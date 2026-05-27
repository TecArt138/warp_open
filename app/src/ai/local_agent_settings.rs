use settings::macros::define_settings_group;
use settings::{SupportedPlatforms, SyncToCloud};
use warp_core::features::FeatureFlag;
use warpui::{AppContext, SingletonEntity as _};

#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    settings_value::SettingsValue,
)]
#[schemars(
    description = "Provider mode for agent conversations.",
    rename_all = "snake_case"
)]
pub enum AgentProviderMode {
    /// Use Warp Cloud (default behavior).
    #[default]
    WarpCloud,
    /// Use a local OpenAI-compatible endpoint.
    Local,
}

define_settings_group!(LocalAgentSettings, settings: [
    agent_provider_mode: AgentProviderModeSetting {
        type: AgentProviderMode,
        default: AgentProviderMode::WarpCloud,
        supported_platforms: SupportedPlatforms::ALL,
        sync_to_cloud: SyncToCloud::Never,
        private: true,
    },
    local_endpoint_url: LocalEndpointUrl {
        type: String,
        default: "http://localhost:11434/v1".to_string(),
        supported_platforms: SupportedPlatforms::ALL,
        sync_to_cloud: SyncToCloud::Never,
        private: true,
    },
    local_model_name: LocalModelName {
        type: String,
        default: "qwen2.5-coder".to_string(),
        supported_platforms: SupportedPlatforms::ALL,
        sync_to_cloud: SyncToCloud::Never,
        private: true,
    },
    local_api_key: LocalApiKey {
        type: String,
        default: String::new(),
        supported_platforms: SupportedPlatforms::ALL,
        sync_to_cloud: SyncToCloud::Never,
        private: true,
    }
]);

impl LocalAgentSettings {
    pub fn is_local_mode_selected(&self) -> bool {
        *self.agent_provider_mode == AgentProviderMode::Local
    }

    pub fn is_local_mode_enabled(app: &AppContext) -> bool {
        FeatureFlag::LocalAgentProvider.is_enabled() && Self::as_ref(app).is_local_mode_selected()
    }
}
