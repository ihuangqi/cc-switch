use bytes::Bytes;
use futures::stream;
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;

// 导入 streaming 模块的函数
use cc_switch_lib::proxy::providers::streaming::{create_anthropic_sse_stream, map_stop_reason};

#[test]
fn test_map_stop_reason_legacy_and_filtered_values() {
    assert_eq!(
        map_stop_reason(Some("function_call")),
        Some("tool_use".to_string())
    );
    assert_eq!(
        map_stop_reason(Some("content_filter")),
        Some("end_turn".to_string())
    );
}

#[tokio::test]
async fn test_streaming_tool_calls_routed_by_index() {
    let input = concat!(
        "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"first_tool\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"second_tool\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":4}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = stream::iter(vec![Ok(Bytes::from(input.as_bytes().to_vec()))]);
    let converted = create_anthropic_sse_stream(upstream);
    let chunks: Vec<_> = converted.collect().await;

    let merged = chunks
        .into_iter()
        .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
        .collect::<String>();

    let events: Vec<Value> = merged
        .split("\n\n")
        .filter_map(|block| {
            let data = block.lines().find_map(|line| line.strip_prefix("data: "))?;
            serde_json::from_str::<Value>(data).ok()
        })
        .collect();

    let mut tool_index_by_call: HashMap<String, u64> = HashMap::new();
    for event in &events {
        if event.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
            && event
                .pointer("/content_block/type")
                .and_then(|v| v.as_str())
                == Some("tool_use")
        {
            if let (Some(call_id), Some(index)) = (
                event.pointer("/content_block/id").and_then(|v| v.as_str()),
                event.get("index").and_then(|v| v.as_u64()),
            ) {
                tool_index_by_call.insert(call_id.to_string(), index);
            }
        }
    }

    assert_eq!(tool_index_by_call.len(), 2);
    assert_ne!(
        tool_index_by_call.get("call_0"),
        tool_index_by_call.get("call_1")
    );

    let deltas: Vec<(u64, String)> = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta")
                && event.pointer("/delta/type").and_then(|v| v.as_str())
                    == Some("input_json_delta")
        })
        .filter_map(|event| {
            let index = event.get("index").and_then(|v| v.as_u64())?;
            let partial_json = event
                .pointer("/delta/partial_json")
                .and_then(|v| v.as_str())?
                .to_string();
            Some((index, partial_json))
        })
        .collect();

    assert_eq!(deltas.len(), 2);
    let second_idx = deltas
        .iter()
        .find_map(|(index, payload)| (payload == "{\"b\":2}").then_some(*index))
        .unwrap();
    let first_idx = deltas
        .iter()
        .find_map(|(index, payload)| (payload == "{\"a\":1}").then_some(*index))
        .unwrap();

    assert_eq!(second_idx, *tool_index_by_call.get("call_1").unwrap());
    assert_eq!(first_idx, *tool_index_by_call.get("call_0").unwrap());

    assert!(events.iter().any(|event| {
        event.get("type").and_then(|v| v.as_str()) == Some("message_delta")
            && event.pointer("/delta/stop_reason").and_then(|v| v.as_str()) == Some("tool_use")
    }));
}

#[tokio::test]
async fn test_streaming_delays_tool_start_until_id_and_name_ready() {
    let input = concat!(
        "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"first_tool\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = stream::iter(vec![Ok(Bytes::from(input.as_bytes().to_vec()))]);
    let converted = create_anthropic_sse_stream(upstream);
    let chunks: Vec<_> = converted.collect().await;
    let merged = chunks
        .into_iter()
        .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
        .collect::<String>();

    let events: Vec<Value> = merged
        .split("\n\n")
        .filter_map(|block| {
            let data = block.lines().find_map(|line| line.strip_prefix("data: "))?;
            serde_json::from_str::<Value>(data).ok()
        })
        .collect();

    let starts: Vec<&Value> = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
                && event
                    .pointer("/content_block/type")
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
        })
        .collect();
    assert_eq!(starts.len(), 1);
    assert_eq!(
        starts[0]
            .pointer("/content_block/id")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        "call_0"
    );
    assert_eq!(
        starts[0]
            .pointer("/content_block/name")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        "first_tool"
    );

    let deltas: Vec<&str> = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta")
                && event.pointer("/delta/type").and_then(|v| v.as_str())
                    == Some("input_json_delta")
        })
        .filter_map(|event| {
            event
                .pointer("/delta/partial_json")
                .and_then(|v| v.as_str())
        })
        .collect();
    assert!(deltas.contains(&"{\"a\":"));
    assert!(deltas.contains(&"1}"));
}

#[tokio::test]
async fn test_streaming_text_single_block_multiple_deltas() {
    // 使用示例.md中的真实数据测试：确保只有一个text block，但有多个delta
    let input = concat!(
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":1,\"total_tokens\":17540},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"我是\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":2,\"total_tokens\":17541},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Cl\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":4,\"total_tokens\":17543},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"aude Code\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":6,\"total_tokens\":17545},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"，是\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":8,\"total_tokens\":17547},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Anthrop\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":10,\"total_tokens\":17549},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ic官方\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":12,\"total_tokens\":17551},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"为Cl\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":14,\"total_tokens\":17553},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"aude开发的\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":16,\"total_tokens\":17555},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"CLI\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":18,\"total_tokens\":17557},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"工具，\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":20,\"total_tokens\":17559},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"运行在\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":22,\"total_tokens\":17561},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Claude\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":24,\"total_tokens\":17563},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\" Agent SDK\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182473,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":26,\"total_tokens\":17565},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"中。\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182474,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":27,\"total_tokens\":17711},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"我\",\"tool_calls\":[]}}]}\n\n",
        "data: {\"id\":\"1a2345ab-1fd5-42d6-8ea5-b0b114a2078b\",\"object\":\"chat.completion.chunk\",\"created\":1774182478,\"model\":\"deepseek-v3.2\",\"usage\":{\"prompt_tokens\":17539,\"completion_tokens\":224,\"total_tokens\":17763},\"choices\":[{\"index\":0,\"finish_reason\":\"stop\",\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[]}}]}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = stream::iter(vec![Ok(Bytes::from(input.as_bytes().to_vec()))]);
    let converted = create_anthropic_sse_stream(upstream);
    let chunks: Vec<_> = converted.collect().await;

    let merged = chunks
        .into_iter()
        .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
        .collect::<String>();

    let events: Vec<Value> = merged
        .split("\n\n")
        .filter_map(|block| {
            let data = block.lines().find_map(|line| line.strip_prefix("data: "))?;
            serde_json::from_str::<Value>(data).ok()
        })
        .collect();

    // 测试：只有一个 content_block_start (type="text")
    let text_block_starts: Vec<&Value> = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
                && event.pointer("/content_block/type").and_then(|v| v.as_str())
                    == Some("text")
        })
        .collect();
    assert_eq!(text_block_starts.len(), 1, "应该只有一个 text 类型的 content_block_start");
    assert_eq!(
        text_block_starts[0].pointer("/content_block/type").and_then(|v| v.as_str()).unwrap(),
        "text"
    );
    assert_eq!(text_block_starts[0].get("index").and_then(|v| as_u64(v)), Some(0));

    // 测试：所有 content_block_delta 都指向同一个 index (0)
    let text_deltas: Vec<(&Value, u64)> = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta")
                && event.pointer("/delta/type").and_then(|v| v.as_str())
                    == Some("text_delta")
        })
        .filter_map(|event| {
            let index = event.get("index").and_then(|v| v.as_u64())?;
            Some((event, index))
        })
        .collect();

    assert!(!text_deltas.is_empty(), "应该有多个 content_block_delta");
    assert!(
        text_deltas.iter().all(|(_, index)| *index == 0),
        "所有 delta 的 index 都应该是 0"
    );

    // 测试：收集所有 delta 内容并验证完整性
    let mut full_text = String::new();
    for (event, _) in &text_deltas {
        if let Some(text) = event.pointer("/delta/text").and_then(|v| v.as_str()) {
            full_text.push_str(text);
        }
    }
    assert!(full_text.contains("我是"));
    assert!(full_text.contains("Claude Code"));
    assert!(full_text.contains("Anthropic"));
    assert!(full_text.contains("官方"));
    assert!(full_text.contains("CLI工具"));

    // 测试：只有一个 content_block_stop (针对 text block)
    let text_block_stops: Vec<u64> = events
        .iter()
        .filter_map(|event| {
            if event.get("type").and_then(|v| v.as_str()) == Some("content_block_stop") {
                event.get("index").and_then(|v| v.as_u64())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(text_block_stops.len(), 1, "应该只有一个 content_block_stop");
    assert_eq!(text_block_stops[0], 0, "content_block_stop 的 index 应该是 0");

    // 测试：验证事件类型顺序正确
    let event_types: Vec<&str> = events
        .iter()
        .filter_map(|event| event.get("type").and_then(|v| v.as_str()))
        .collect();

    // 应该是：message_start -> content_block_start -> 多个 content_block_delta -> content_block_stop -> message_delta -> message_stop
    assert_eq!(event_types[0], "message_start");
    assert_eq!(event_types[1], "content_block_start");
    assert!(event_types.iter().any(|t| *t == "content_block_delta"));
    assert_eq!(
        event_types.last().unwrap(),
        &"message_stop"
    );

    println!("✅ 测试通过：单个text block，{}个delta", text_deltas.len());
}

fn as_u64(v: &Value) -> Option<u64> {
    v.as_u64()
}