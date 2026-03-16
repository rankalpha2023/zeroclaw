//! Custom Provider 工具调用修复 - 单元测试
//!
//! 测试修改后的 Custom Provider 是否正确设置 native_tool_calling = false

use zeroclaw::providers::create_provider_with_options;
use zeroclaw::providers::ProviderRuntimeOptions;

/// 测试 Custom Provider 的 native_tool_calling 被设置为 false
#[test]
fn test_custom_provider_native_tool_calling_disabled() {
    // 创建 Custom Provider
    let options = ProviderRuntimeOptions::default();
    let provider = create_provider_with_options(
        "custom:http://localhost:3450",
        Some("test-key"),
        &options,
    ).expect("Failed to create custom provider");
    
    // 验证：native_tool_calling 应该为 false
    assert!(
        !provider.supports_native_tools(),
        "Custom provider should have native_tool_calling = false"
    );
}

/// 测试 Custom Provider 的 vision 功能仍然启用
#[test]
fn test_custom_provider_vision_enabled() {
    // 创建 Custom Provider
    let options = ProviderRuntimeOptions::default();
    let provider = create_provider_with_options(
        "custom:http://localhost:3450",
        Some("test-key"),
        &options,
    ).expect("Failed to create custom provider");
    
    // 验证：vision 应该为 true
    let capabilities = provider.capabilities();
    assert!(
        capabilities.vision,
        "Custom provider should have vision enabled"
    );
}

/// 测试不同 URL 格式的 Custom Provider
#[test]
fn test_custom_provider_with_different_urls() {
    let test_urls = vec![
        "custom:http://localhost:3450",
        "custom:http://127.0.0.1:8080",
        "custom:https://api.example.com/v1",
    ];
    
    let options = ProviderRuntimeOptions::default();
    
    for url in test_urls {
        let provider = create_provider_with_options(
            url,
            Some("test-key"),
            &options,
        ).expect(&format!("Failed to create custom provider for {}", url));
        
        // 验证每个 URL 的 provider 都正确设置
        assert!(
            !provider.supports_native_tools(),
            "Custom provider ({}) should have native_tool_calling = false",
            url
        );
        
        assert!(
            provider.capabilities().vision,
            "Custom provider ({}) should have vision enabled",
            url
        );
    }
}

/// 测试没有 API Key 的 Custom Provider
#[test]
fn test_custom_provider_without_api_key() {
    let options = ProviderRuntimeOptions::default();
    let provider = create_provider_with_options(
        "custom:http://localhost:3450",
        None,  // 没有 API Key
        &options,
    ).expect("Failed to create custom provider without API key");
    
    // 验证：即使没有 API Key，属性也应该正确设置
    assert!(
        !provider.supports_native_tools(),
        "Custom provider without API key should have native_tool_calling = false"
    );
    
    assert!(
        provider.capabilities().vision,
        "Custom provider without API key should have vision enabled"
    );
}

/// 测试其他 Provider 不受影响（回归测试）
#[test]
fn test_other_providers_not_affected() {
    // 测试几个主要的 Provider，确保它们不受 Custom Provider 修改的影响
    let test_providers = vec![
        ("openrouter", Some("test-key")),
        ("openai", Some("test-key")),
        ("anthropic", Some("test-key")),
    ];
    
    let options = ProviderRuntimeOptions::default();
    
    for (provider_name, api_key) in test_providers {
        let provider = create_provider_with_options(
            provider_name,
            api_key,
            &options,
        ).expect(&format!("Failed to create {} provider", provider_name));
        
        // 这些 Provider 应该保持原有的行为
        // （大多数应该支持原生工具调用）
        assert!(
            provider.supports_native_tools(),
            "{} provider should support native tool calling",
            provider_name
        );
    }
}
