// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! AWS SigV4 request signing for AWS-native Iceberg REST endpoints (S3
//! Tables, Glue), as Iceberg Java's `SigV4AuthManager`: selected by
//! `rest.auth.type=sigv4`, or by the older `rest.sigv4-enabled=true`.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use async_trait::async_trait;
use aws_credential_types::provider::SharedCredentialsProvider;
use iceberg::{Error, ErrorKind, Result};
use tokio::sync::OnceCell;

use super::{AuthManager, AuthSession, HttpRequest};
use crate::client::HttpClient;

/// The legacy property enabling SigV4 signing, without `rest.auth.type`.
pub const REST_SIGV4_ENABLED: &str = "rest.sigv4-enabled";
/// The AWS region requests are signed for; falls back to `s3.region`, then
/// `us-east-1`.
pub const REST_SIGNING_REGION: &str = "rest.signing-region";
/// The AWS service requests are signed for (`s3tables` for S3 Tables,
/// `glue` for Glue); defaults to `execute-api`.
pub const REST_SIGNING_NAME: &str = "rest.signing-name";

/// Whether `props` turn on SigV4 through the legacy `rest.sigv4-enabled`.
pub(crate) fn sigv4_enabled(props: &HashMap<String, String>) -> bool {
    props
        .get(REST_SIGV4_ENABLED)
        .is_some_and(|v| v.eq_ignore_ascii_case("true"))
}

/// [`AuthManager`] signing every request with AWS SigV4. Credentials come
/// from the standard AWS chain (environment, ECS/EKS task role, EC2 IMDS,
/// shared config), loaded on first use and kept for the catalog's lifetime.
pub struct SigV4AuthManager {
    region: String,
    name: String,
    credentials: Arc<OnceCell<SharedCredentialsProvider>>,
}

impl Debug for SigV4AuthManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigV4AuthManager")
            .field("region", &self.region)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl SigV4AuthManager {
    /// A manager signing for the region and service `props` name.
    pub fn from_props(props: &HashMap<String, String>) -> Self {
        let (region, name) = signing(props, None);
        Self {
            region,
            name,
            credentials: Arc::new(OnceCell::new()),
        }
    }

    /// The same, signing with `provider`'s credentials instead of the
    /// default chain's.
    pub fn with_credentials_provider(self, provider: SharedCredentialsProvider) -> Self {
        Self {
            credentials: Arc::new(OnceCell::new_with(Some(provider))),
            ..self
        }
    }

    fn session(&self, region: String, name: String) -> SigV4Session {
        SigV4Session {
            region,
            name,
            credentials: self.credentials.clone(),
        }
    }
}

/// The region and service `props` sign for, or `fallback`'s.
fn signing(props: &HashMap<String, String>, fallback: Option<(&str, &str)>) -> (String, String) {
    let region = props
        .get(REST_SIGNING_REGION)
        .or_else(|| props.get("s3.region"))
        .cloned()
        .or_else(|| fallback.map(|(region, _)| region.to_string()))
        .unwrap_or_else(|| "us-east-1".to_string());
    let name = props
        .get(REST_SIGNING_NAME)
        .cloned()
        .or_else(|| fallback.map(|(_, name)| name.to_string()))
        .unwrap_or_else(|| "execute-api".to_string());
    (region, name)
}

#[async_trait]
impl AuthManager for SigV4AuthManager {
    async fn init_session(
        &self,
        _client: &HttpClient,
        _props: &HashMap<String, String>,
    ) -> Result<Box<dyn AuthSession>> {
        Ok(Box::new(self.session(self.region.clone(), self.name.clone())))
    }

    async fn catalog_session(
        &self,
        _client: &HttpClient,
        props: &HashMap<String, String>,
    ) -> Result<Arc<dyn AuthSession>> {
        // The server's config may name the region or service.
        let (region, name) = signing(props, Some((&self.region, &self.name)));
        Ok(Arc::new(self.session(region, name)))
    }
}

/// [`AuthSession`] signing every request with AWS SigV4.
struct SigV4Session {
    region: String,
    name: String,
    credentials: Arc<OnceCell<SharedCredentialsProvider>>,
}

impl Debug for SigV4Session {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigV4Session")
            .field("region", &self.region)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl SigV4Session {
    async fn provider(&self) -> Result<&SharedCredentialsProvider> {
        self.credentials
            .get_or_try_init(|| async {
                let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
                cfg.credentials_provider().ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "no AWS credentials provider for SigV4 signing",
                    )
                })
            })
            .await
    }
}

#[async_trait]
impl AuthSession for SigV4Session {
    /// Adds `authorization`, `x-amz-date` and `x-amz-content-sha256`,
    /// computed over the request's method, URL, headers and body as they
    /// stand: the client sets every other header first.
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        use aws_credential_types::provider::ProvideCredentials;
        use aws_sigv4::http_request::{
            PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings, sign,
        };
        use aws_sigv4::sign::v4;

        let creds = self
            .provider()
            .await?
            .provide_credentials()
            .await
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "load AWS credentials").with_source(e)
            })?;
        let identity: aws_smithy_runtime_api::client::identity::Identity = creds.into();

        let mut settings = SigningSettings::default();
        // AWS-native catalogs (S3 Tables, Glue) expect the payload hash.
        settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(&self.name)
            .time(std::time::SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| Error::new(ErrorKind::Unexpected, "build SigV4 params").with_source(e))?
            .into();

        let body = request.body().as_bytes().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "SigV4 can't sign a request with a streaming body",
            )
        })?;
        let method = request.method().as_str().to_string();
        let uri = request.url_str().to_string();
        let headers: Vec<(String, String)> = request
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let signable = SignableRequest::new(
            &method,
            &uri,
            headers.iter().map(|(k, v)| (k.as_str(), v.as_str())),
            SignableBody::Bytes(body),
        )
        .map_err(|e| Error::new(ErrorKind::Unexpected, "build signable request").with_source(e))?;
        let (instructions, _signature) = sign(signable, &params)
            .map_err(|e| Error::new(ErrorKind::Unexpected, "SigV4 sign").with_source(e))?
            .into_parts();

        // Apply the signing headers to a throwaway request mirroring this
        // one, then copy them over.
        let mut mirror = http::Request::builder()
            .method(request.method().clone())
            .uri(uri.as_str());
        for (k, v) in request.headers().iter() {
            mirror = mirror.header(k, v);
        }
        let mut mirror = mirror.body(()).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "build http request for signing").with_source(e)
        })?;
        instructions.apply_to_request_http1x(&mut mirror);
        for (k, v) in mirror.headers() {
            request.headers_mut().insert(k.clone(), v.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use aws_credential_types::Credentials;
    use reqwest::{Client, Method};

    use super::*;

    fn manager(props: &[(&str, &str)]) -> SigV4AuthManager {
        let props = props
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        SigV4AuthManager::from_props(&props).with_credentials_provider(
            SharedCredentialsProvider::new(Credentials::new(
                "AKID", "SECRET", None, None, "test",
            )),
        )
    }

    async fn signed(manager: &SigV4AuthManager, catalog_props: &[(&str, &str)]) -> HttpRequest {
        let config = crate::RestCatalogConfig::builder()
            .uri("http://localhost".to_string())
            .build();
        let client = HttpClient::new(&config).unwrap();
        let props = catalog_props
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let session = manager.catalog_session(&client, &props).await.unwrap();
        let mut request = HttpRequest::new(
            Client::new()
                .request(Method::POST, "https://s3tables.eu-west-1.amazonaws.com/iceberg/v1/config")
                .body("{\"a\":1}")
                .build()
                .unwrap(),
        );
        session.authenticate(&mut request).await.unwrap();
        request
    }

    fn header(request: &HttpRequest, name: &str) -> String {
        request
            .headers()
            .get(name)
            .map(|v| v.to_str().unwrap().to_string())
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn test_signs_for_the_configured_region_and_service() {
        let manager = manager(&[
            (REST_SIGV4_ENABLED, "true"),
            (REST_SIGNING_REGION, "eu-west-1"),
            (REST_SIGNING_NAME, "s3tables"),
        ]);
        let request = signed(&manager, &[]).await;
        let authorization = header(&request, "authorization");
        assert!(
            authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKID/"),
            "{authorization}"
        );
        assert!(
            authorization.contains("/eu-west-1/s3tables/aws4_request"),
            "{authorization}"
        );
        assert!(!header(&request, "x-amz-date").is_empty());
        assert_eq!(header(&request, "x-amz-content-sha256").len(), 64);
    }

    #[tokio::test]
    async fn test_server_config_names_the_region_and_service() {
        let manager = manager(&[(REST_SIGNING_REGION, "eu-west-1")]);
        let request = signed(&manager, &[(REST_SIGNING_NAME, "glue")]).await;
        let authorization = header(&request, "authorization");
        assert!(
            authorization.contains("/eu-west-1/glue/aws4_request"),
            "{authorization}"
        );
    }

    #[tokio::test]
    async fn test_region_falls_back_to_s3_region_then_us_east_1() {
        let request = signed(&manager(&[("s3.region", "ap-southeast-1")]), &[]).await;
        assert!(header(&request, "authorization").contains("/ap-southeast-1/execute-api/"));
        let request = signed(&manager(&[]), &[]).await;
        assert!(header(&request, "authorization").contains("/us-east-1/execute-api/"));
    }

    #[test]
    fn test_legacy_property_enables_signing() {
        let props = |v: &str| HashMap::from([(REST_SIGV4_ENABLED.to_string(), v.to_string())]);
        assert!(sigv4_enabled(&props("true")));
        assert!(sigv4_enabled(&props("TRUE")));
        assert!(!sigv4_enabled(&props("false")));
        assert!(!sigv4_enabled(&HashMap::new()));
    }
}
