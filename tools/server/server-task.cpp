#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"
#include "src/llama-ext.h"
#define XXH_STATIC_LINKING_ONLY
#include "hash/xxhash/xxhash.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_loop_guard",      common_reasoning_loop_guard_mode_name(reasoning_loop_guard.mode)},
            {"reasoning_loop_min_tokens", reasoning_loop_guard.min_reasoning_tokens},
            {"reasoning_loop_window",     reasoning_loop_guard.window_tokens},
            {"reasoning_loop_max_period", reasoning_loop_guard.max_period},
            {"reasoning_loop_min_coverage", reasoning_loop_guard.min_repeated_coverage},
            {"reasoning_loop_check_interval", reasoning_loop_guard.check_interval},
            {"reasoning_loop_interventions", reasoning_loop_guard.interventions_max},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_loop_guard",      common_reasoning_loop_guard_mode_name(reasoning_loop_guard.mode)},
        {"reasoning_loop_min_tokens", reasoning_loop_guard.min_reasoning_tokens},
        {"reasoning_loop_window",     reasoning_loop_guard.window_tokens},
        {"reasoning_loop_max_period", reasoning_loop_guard.max_period},
        {"reasoning_loop_min_coverage", reasoning_loop_guard.min_repeated_coverage},
        {"reasoning_loop_check_interval", reasoning_loop_guard.check_interval},
        {"reasoning_loop_interventions", reasoning_loop_guard.interventions_max},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stop_detail",         stop_detail},
        {"reasoning_tokens",    reasoning_output_tokens},
        {"visible_completion_tokens", visible_output_tokens},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (loop_guard_event.triggered) {
        res["loop_guard"] = json {
            {"triggered", true},
            {"region", loop_guard_event.region},
            {"detector", loop_guard_event.detector},
            {"period", loop_guard_event.period},
            {"coverage", loop_guard_event.coverage},
            {"score", loop_guard_event.score},
            {"interventions", loop_guard_event.interventions},
            {"action", loop_guard_event.action},
            {"decoded_token_index", loop_guard_event.decoded_token_index},
            {"token", loop_guard_event.token},
            {"token_piece", loop_guard_event.token_piece},
            {"reason", loop_guard_event.reason},
        };
    }
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    json usage = json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
    usage["completion_tokens_details"] = json {
        {"reasoning_tokens", reasoning_output_tokens},
        {"visible_tokens", visible_output_tokens},
    };
    return usage;
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        }, {
            "prompt_cache_admission_attempts_total",
            "Total immutable RAM prompt-cache admission attempts",
            (double) metrics.prompt_cache_admission_attempts
        }, {
            "prompt_cache_admission_successes_total",
            "Total immutable RAM prompt-cache admissions",
            (double) metrics.prompt_cache_admission_successes
        }, {
            "prompt_cache_admission_failures_total",
            "Total rejected RAM prompt-cache admissions",
            (double) metrics.prompt_cache_admission_failures
        }, {
            "prompt_cache_restore_attempts_total",
            "Total transactional RAM prompt-cache restore attempts",
            (double) metrics.prompt_cache_restore_attempts
        }, {
            "prompt_cache_restore_successes_total",
            "Total committed RAM prompt-cache restores",
            (double) metrics.prompt_cache_restore_successes
        }, {
            "prompt_cache_restore_failures_total",
            "Total aborted RAM prompt-cache restores",
            (double) metrics.prompt_cache_restore_failures
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        }, {
            "kv_tail_requested_tokens",
            "Configured exact-tail tokens currently requested across server slots and cache groups",
            (double) metrics.kv_tail_requested
        }, {
            "kv_tail_exact_tokens",
            "Exact-tail tokens currently covered across server slots and cache groups",
            (double) metrics.kv_tail_exact
        }, {
            "kv_tail_complete_groups",
            "Server slot cache groups with complete exact-tail coverage",
            (double) metrics.kv_tail_complete_groups
        }, {
            "kv_tail_partial_groups",
            "Server slot cache groups with partial exact-tail coverage",
            (double) metrics.kv_tail_partial_groups
        }, {
            "kv_tail_none_groups",
            "Server slot cache groups with no exact-tail coverage",
            (double) metrics.kv_tail_none_groups
        }, {
            "kv_tail_degraded_sequences",
            "Server slots reporting an explicit exact-tail degradation reason",
            (double) metrics.kv_tail_degraded_sequences
        }, {
            "prompt_cache_accounted_bytes",
            "Serialized RAM prompt-cache payload bytes, excluding container and allocator overhead",
            (double) metrics.prompt_cache_accounted_bytes
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

size_t server_prompt_cache::accounted_size() const {
    size_t res = 0;
    std::unordered_set<const void *> checkpoint_storage;

    for (const auto & state : states) {
        res += state.data.size();
        for (const auto & checkpoint : state.prompt.checkpoints) {
            if (!checkpoint->data_tgt.empty() &&
                    checkpoint_storage.insert(checkpoint->data_tgt.data()).second) {
                res += checkpoint->data_tgt.size();
            }
            if (!checkpoint->data_dft.empty() &&
                    checkpoint_storage.insert(checkpoint->data_dft.data()).second) {
                res += checkpoint->data_dft.size();
            }
            res += checkpoint->data_spec.size();
        }
    }
    return res;
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size(const server_prompt_cache_state * extra) const {
    size_t res = transient_bytes;
    bool found_extra = false;
    std::unordered_set<const common_prompt_checkpoint *> checkpoints;
    auto add = [&](const server_prompt_cache_state & state) {
        res += state.private_size();
        for (const auto & c : state.prompt.checkpoints) {
            if (checkpoints.insert(c.get()).second) { res += c->size() + 2*sizeof(void *); }
        }
    };
    for (const auto & state : states) {
        add(state);
        found_extra = found_extra || &state == extra;
    }
    if (extra && !found_extra) { add(*extra); }
    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

uint64_t server_prompt_cache_state::digest() const {
    XXH64_state_t hash;
    XXH64_reset(&hash, 0);
    auto add = [&](const void * ptr, size_t n) { XXH64_update(&hash, ptr, n); };
    auto bytes = [&](const std::vector<uint8_t> & v) {
        const size_t n = v.size();
        add(&n, sizeof(n));
        add(v.data(), n);
    };
    add(&model_instance, sizeof(model_instance));
    add(layout_tgt.data(), layout_tgt.size());
    add(layout_dft.data(), layout_dft.size());
    add(&pos_tgt, sizeof(pos_tgt));
    add(&pos_dft, sizeof(pos_dft));
    const uint64_t tokens_hash = prompt.tokens.cache_digest();
    add(&tokens_hash, sizeof(tokens_hash));
    bytes(data.main); bytes(data.drft); bytes(data.spec);
    for (const auto & c : prompt.checkpoints) {
        add(&c->instance_tgt, sizeof(c->instance_tgt)); add(&c->instance_dft, sizeof(c->instance_dft));
        add(c->layout_tgt.data(), c->layout_tgt.size()); add(c->layout_dft.data(), c->layout_dft.size());
        add(c->attention_tgt.data(), sizeof(c->attention_tgt)); add(c->attention_dft.data(), sizeof(c->attention_dft));
        add(c->retained_tgt.data(), sizeof(c->retained_tgt)); add(c->retained_dft.data(), sizeof(c->retained_dft));
        add(&c->retained_count_tgt, sizeof(c->retained_count_tgt)); add(&c->retained_count_dft, sizeof(c->retained_count_dft));
        add(&c->draft_base_valid, sizeof(c->draft_base_valid)); add(&c->id_task, sizeof(c->id_task));
        add(&c->n_tokens, sizeof(c->n_tokens));
        add(&c->pos_min, sizeof(c->pos_min)); add(&c->pos_max, sizeof(c->pos_max));
        add(&c->flags_tgt, sizeof(c->flags_tgt)); add(&c->flags_dft, sizeof(c->flags_dft));
        bytes(c->data_tgt); bytes(c->data_dft); bytes(c->data_spec);
    }
    return XXH64_digest(&hash);
}

bool server_prompt_cache::make_room(size_t bytes, const server_prompt_cache_state * incoming,
        const server_prompt_cache_state * keep) {
    if (!limit_size) { return true; }
    for (;;) {
        const size_t used = size(incoming);
        if (used <= limit_size && bytes <= limit_size - used) { return true; }
        const auto victim = std::find_if(states.begin(), states.end(), [&](const auto & state) { return &state != keep; });
        if (victim == states.end()) {
            last_reason = "snapshot and temporary buffers exceed RAM cache budget";
            return false;
        }
        SRV_WRN("RAM cache eviction: %d tokens, %.3f MiB, draft=%d\n",
                victim->prompt.n_tokens(), victim->size() / (1024.0 * 1024.0), victim->has_draft());
        states.erase(victim);
        ++evictions;
    }
}

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io) {
    if (!io.prepare || !io.commit) {
        return { false, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_INVALID_IO };
    }
    if (io.restore_target && target.size == 0) {
        return { false, true, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE };
    }
    if (io.restore_draft && draft.size == 0) {
        return { false, true, SERVER_PROMPT_STATE_DRAFT, SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE };
    }

    const auto prepare = [&](bool enabled, server_prompt_state_kind kind, server_prompt_state_view state) {
        if (enabled && !io.prepare(kind, state)) {
            return server_prompt_restore_result {
                false, true, kind, SERVER_PROMPT_RESTORE_PREPARE_REJECTED
            };
        }
        return server_prompt_restore_result {
            true, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_NONE
        };
    };
    for (const auto & step : {
            std::pair { io.restore_target, SERVER_PROMPT_STATE_MAIN },
            std::pair { io.restore_draft, SERVER_PROMPT_STATE_DRAFT },
            std::pair { io.restore_speculative, SERVER_PROMPT_STATE_SPECULATIVE } }) {
        const server_prompt_state_view state = step.second == SERVER_PROMPT_STATE_MAIN ? target :
                step.second == SERVER_PROMPT_STATE_DRAFT ? draft : speculative;
        const auto result = prepare(step.first, step.second, state);
        if (!result.success) {
            return result;
        }
    }

    // Speculative apply is prepared and no-fail. Memory commits likewise only
    // publish already-validated backend writes and metadata.
    if (io.restore_speculative) {
        io.commit(SERVER_PROMPT_STATE_SPECULATIVE);
    }
    if (io.restore_target) {
        io.commit(SERVER_PROMPT_STATE_MAIN);
    }
    if (io.restore_draft) {
        io.commit(SERVER_PROMPT_STATE_DRAFT);
    }
    return { true, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_NONE };
}

bool server_prompt_restore_transaction(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io) {
    return server_prompt_restore_transaction_diagnostic(target, draft, speculative, io).success;
}

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        llama_context * target,
        llama_context * draft,
        common_speculative * speculative,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        server_prompt_state_view target_state,
        server_prompt_state_view draft_state,
        server_prompt_state_view speculative_state,
        bool restore_target,
        bool restore_draft,
        bool restore_speculative) {
    using memory_plan_ptr = std::unique_ptr<
            llama_state_seq_restore_plan,
            decltype(&llama_state_seq_restore_plan_free)>;
    using speculative_plan_ptr = std::unique_ptr<
            common_speculative_state_restore_plan,
            decltype(&common_speculative_state_restore_plan_free)>;

    memory_plan_ptr target_plan(nullptr, llama_state_seq_restore_plan_free);
    memory_plan_ptr draft_plan(nullptr, llama_state_seq_restore_plan_free);
    speculative_plan_ptr speculative_plan(nullptr, common_speculative_state_restore_plan_free);

    server_prompt_restore_transaction_io io {
        /*.restore_target =*/ restore_target,
        /*.restore_draft =*/ restore_draft,
        /*.restore_speculative =*/ restore_speculative,
        /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view state) {
            if (kind == SERVER_PROMPT_STATE_SPECULATIVE) {
                speculative_plan.reset(common_speculative_prepare_state(
                        speculative, seq_id, state.data, state.size));
                return speculative_plan != nullptr;
            }

            llama_context * ctx = kind == SERVER_PROMPT_STATE_MAIN ? target : draft;
            memory_plan_ptr & plan = kind == SERVER_PROMPT_STATE_MAIN ? target_plan : draft_plan;
            if (ctx == nullptr) {
                return false;
            }
            plan.reset(llama_state_seq_prepare_data_ext(
                    ctx, state.data, state.size, seq_id, flags));
            return plan != nullptr;
        },
        /*.commit =*/ [&](server_prompt_state_kind kind) {
            if (kind == SERVER_PROMPT_STATE_SPECULATIVE) {
                common_speculative_state_restore_plan_commit(speculative_plan.get());
                return;
            }
            memory_plan_ptr & plan = kind == SERVER_PROMPT_STATE_MAIN ? target_plan : draft_plan;
            const size_t expected = kind == SERVER_PROMPT_STATE_MAIN ? target_state.size : draft_state.size;
            GGML_ASSERT(llama_state_seq_restore_plan_commit(plan.get()) == expected);
        },
    };
    return server_prompt_restore_transaction_diagnostic(
            target_state, draft_state, speculative_state, io);
}

bool server_prompt_cache::reserve_transient(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - transient_bytes) {
        last_reason = "transient buffers exceed RAM cache budget";
        return false;
    }
    if (limit_size && !make_room(bytes)) {
        return false;
    }
    transient_bytes += bytes;
    return true;
}

bool server_prompt_restore_transaction(
        llama_context * target,
        llama_context * draft,
        common_speculative * speculative,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        server_prompt_state_view target_state,
        server_prompt_state_view draft_state,
        server_prompt_state_view speculative_state,
        bool restore_target,
        bool restore_draft,
        bool restore_speculative) {
    return server_prompt_restore_transaction_diagnostic(
            target, draft, speculative, seq_id, flags,
            target_state, draft_state, speculative_state,
            restore_target, restore_draft, restore_speculative).success;
}

void server_prompt_cache::release_transient(size_t bytes) {
    transient_bytes = bytes >= transient_bytes ? 0 : transient_bytes - bytes;
}

bool server_prompt_cache::save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
        common_speculative * spec, llama_seq_id id_slot) {
    last_reason.clear();
    if (prompt.tokens.empty()) {
        last_reason = "empty prompt";
        return false;
    }
    llama_synchronize(ctx_tgt);
    if (ctx_dft) { llama_synchronize(ctx_dft); }
    const auto * model = llama_get_model(ctx_tgt);
    const auto info = llama_model_mtp_weights_get_info(model);
    const auto pos_tgt = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id_slot);
    const auto pos_dft = ctx_dft ? llama_memory_seq_pos_max(llama_get_memory(ctx_dft), id_slot) : -1;
    const size_t evaluated = prompt.tokens.size_up_to_pos(pos_tgt + 1);
    if (pos_tgt < 0 || !evaluated || prompt.tokens.pos_next(evaluated) != pos_tgt + 1 ||
            (info.managed && prompt.tokens.has_mtmd)) {
        last_reason = "prompt tokens do not cover evaluated positions";
        return false;
    }
    if (info.managed && ctx_dft && !common_speculative_is_ready(spec, id_slot, pos_tgt + 1)) {
        last_reason = "MTP state is not aligned with evaluated target";
        return false;
    }
    const size_t evictions_before = evictions;
    auto failed = [&](const char * reason) {
        last_reason = reason;
        if (evictions != evictions_before) {
            SRV_WRN("snapshot save failed after %zu pressure evictions: %s\n", evictions - evictions_before, reason);
        }
        return false;
    };
    try {
        server_prompt_cache_state candidate;
        candidate.model = model;
        candidate.model_instance = info.model_instance;
        candidate.layout_tgt = common_prompt_cache_layout(ctx_tgt);
        candidate.layout_dft = ctx_dft ? common_prompt_cache_layout(ctx_dft) : "";
        candidate.pos_tgt = pos_tgt;
        candidate.pos_dft = pos_dft;
        common_speculative_get_state(spec, id_slot, candidate.data.spec);
        const size_t n_tgt = llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        for (const auto & c : prompt.checkpoints) {
            if (c->host_only() && c->compatible_tgt(ctx_tgt) && c->n_tokens <= (int64_t) evaluated && c->pos_max <= pos_tgt) {
                candidate.prompt.checkpoints.push_back(c);
            } else {
                SRV_WRN("%s", "discarding non-host, incompatible or unevaluated prompt checkpoint\n");
            }
        }
        const size_t tokens_copy = prompt.tokens.cache_size(true);
        const size_t restore_copy = tokens_copy + candidate.prompt.checkpoints.size()*
            (sizeof(std::shared_ptr<const common_prompt_checkpoint>) + 2*sizeof(void *));
        const size_t future_bytes = n_tgt + n_dft + tokens_copy;
        const size_t workspace = std::max(restore_copy, prompt.tokens.digest_workspace());
        if (!n_tgt) { return failed("empty target snapshot"); }
        if (limit_size && (candidate.size() > limit_size ||
                future_bytes > limit_size - candidate.size() || workspace > limit_size - candidate.size() - future_bytes)) {
            return failed("snapshot plus restore workspace exceeds RAM cache budget");
        }
        if (!make_room(future_bytes + prompt.tokens.digest_workspace(), &candidate)) { return failed("snapshot reservation failed"); }
        candidate.prompt.tokens = prompt.tokens.clone_for_cache();
        candidate.prompt.tokens.keep_first(evaluated);
        candidate.data.main.resize(n_tgt);
        candidate.data.drft.resize(n_dft);
        if (llama_state_seq_get_data_ext(ctx_tgt, candidate.data.main.data(), n_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) != n_tgt ||
                (ctx_dft && llama_state_seq_get_data_ext(ctx_dft, candidate.data.drft.data(), n_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) != n_dft)) {
            return failed("snapshot serialization failed");
        }
        candidate.checksum = candidate.digest();
        // Allocate the new list node before removing any replaced source.
        states.push_back(std::move(candidate));
        const auto inserted = std::prev(states.end());
        const auto & published = *inserted;
        // Only identical variants supersede each other. A longer target-only entry cannot replace MTP state.
        for (auto it = states.begin(); it != inserted;) {
            if (it->model == model && it->model_instance == info.model_instance &&
                    it->layout_tgt == published.layout_tgt && it->layout_dft == published.layout_dft &&
                    it->data.spec.empty() == published.data.spec.empty() &&
                    it->prompt.tokens.size() == evaluated &&
                    it->prompt.tokens.get_common_prefix(published.prompt.tokens) == evaluated) {
                it = states.erase(it);
            } else { ++it; }
        }
        return true;
    } catch (const std::exception & e) {
        return failed(e.what());
    }
}

static void prompt_cache_clear(llama_context * tgt, llama_context * dft, common_speculative * spec, llama_seq_id seq) {
    llama_memory_seq_rm(llama_get_memory(tgt), seq, -1, -1);
    if (dft) { llama_memory_seq_rm(llama_get_memory(dft), seq, -1, -1); }
    common_speculative_set_state(spec, seq, {});
}

server_prompt_cache_result server_prompt_cache::restore(server_prompt & prompt, const server_prompt_cache_state & state,
        llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id id_slot) {
    return restore_impl(prompt, state, ctx_tgt, ctx_dft, spec, id_slot, true, false);
}

server_prompt_cache_result server_prompt_cache::restore_impl(server_prompt & prompt, const server_prompt_cache_state & state,
        llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id id_slot,
        bool copy_prompt, bool allow_bootstrap) {
    const auto info = llama_model_mtp_weights_get_info(llama_get_model(ctx_tgt));
    restore_invalid = false;
    auto miss = [&](const char * reason, bool invalid = false) {
        last_reason = reason;
        restore_invalid = invalid;
        return server_prompt_cache_result::miss;
    };
    if (state.model != llama_get_model(ctx_tgt) || state.model_instance != info.model_instance) {
        return miss("snapshot model differs");
    }
    const std::string target_layout = common_prompt_cache_layout(ctx_tgt);
    const bool converted_mode =
        !common_prompt_cache_layout_reusable(state.layout_tgt, target_layout) &&
        common_prompt_cache_layout_convertible(state.layout_tgt, target_layout);
    if ((!common_prompt_cache_layout_reusable(state.layout_tgt, target_layout) && !converted_mode) ||
            (!converted_mode && ctx_dft && state.has_draft() &&
             !common_prompt_cache_layout_reusable(state.layout_dft, common_prompt_cache_layout(ctx_dft)))) {
        return miss("snapshot model or attention layout differs");
    }
    if (state.pos_tgt < 0 || state.prompt.tokens.empty() || state.pos_tgt >= (llama_pos) llama_n_ctx_seq(ctx_tgt) ||
            state.prompt.tokens.size() > llama_n_ctx_seq(ctx_tgt) ||
            (!converted_mode && ctx_dft && state.has_draft() && state.pos_dft >= (llama_pos) llama_n_ctx_seq(ctx_dft))) {
        return miss("snapshot does not fit destination context");
    }
    if (state.pos_dft < -1 || state.prompt.tokens.pos_next() != state.pos_tgt + 1 ||
            (info.managed && state.prompt.tokens.has_mtmd)) {
        return miss("snapshot tokens and positions disagree", true);
    }
    try {
        const size_t scratch = state.prompt.tokens.digest_workspace();
        if (limit_size && (state.size() > limit_size || scratch > limit_size - state.size())) {
            return miss("snapshot integrity workspace exceeds RAM cache budget");
        }
        if (scratch && !make_room(scratch, &state, &state)) {
            return miss("snapshot integrity workspace exceeds RAM cache budget");
        }
        if (state.data.main.empty() || state.digest() != state.checksum) {
            return miss("snapshot integrity check failed", true);
        }
    } catch (const std::bad_alloc &) { return miss("snapshot integrity allocation failed"); }
      catch (const std::exception & e) { return miss(e.what(), true); }
    const bool bootstrap_only = info.managed && ctx_dft &&
        (converted_mode || !state.has_draft() || state.data.spec.empty());
    if (bootstrap_only && !allow_bootstrap) {
        last_reason = "target-only snapshot requires MTP bootstrap";
        return server_prompt_cache_result::needs_bootstrap;
    }
    server_prompt candidate;
    try {
        // Checkpoints share immutable host buffers; only the token list is copied before live mutation.
        if (copy_prompt) {
            const size_t copy_bytes = state.prompt.tokens.cache_size(true) + state.prompt.checkpoints.size()*
                (sizeof(std::shared_ptr<const common_prompt_checkpoint>) + 2*sizeof(void *));
            if (limit_size && (state.size() > limit_size || copy_bytes > limit_size - state.size())) {
                return miss("restore prompt copy exceeds RAM cache budget");
            }
            if (!make_room(copy_bytes, &state, &state)) {
                return miss("restore prompt copy exceeds RAM cache budget");
            }
            candidate = state.prompt.clone();
        }
    } catch (const std::bad_alloc &) { return miss("restore allocation failed"); }
    prompt_cache_clear(ctx_tgt, ctx_dft, spec, id_slot);
    bool target_ok = false;
    if (converted_mode) {
        // The cached payload is a q4 sequence state (data form); the destination
        // needs the KVarN layout. Convert through the streamed q4->KVarN path into
        // a private file and load it straight into the destination context.
        try {
            const auto & tokens = state.prompt.tokens.get_tokens();
            const auto base = std::filesystem::temp_directory_path();
            const auto dir = base / (".adaptive-load-convert-" +
                                     std::to_string(ggml_time_us()));
            if (!std::filesystem::create_directory(dir)) {
                throw std::runtime_error("cannot create private conversion directory");
            }
            struct cleanup {
                std::filesystem::path dir;
                ~cleanup() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
            } guard{dir};
            std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                         std::filesystem::perm_options::replace);
            const std::string path = (dir / "state.bin").string();
            size_t count = 0;
            std::vector<llama_token> restored(tokens.size());
            const auto layout = common_json::parse(state.layout_tgt);
            const size_t written = llama_state_seq_convert_data_rotated(
                ctx_tgt, state.data.main.data(), state.data.main.size(), 0,
                tokens.data(), tokens.size(), path.c_str(),
                restored.data(), restored.size(), &count,
                layout.at("rotation_k").get<int32_t>(), layout.at("rotation_v").get<int32_t>());
            if (written && count == tokens.size() && restored == tokens) {
                XXH64_state_t hash;
                XXH64_reset(&hash, 0);
                std::ifstream input(path, std::ios::binary);
                std::vector<char> buffer(1024 * 1024);
                for (size_t left = written; left;) {
                    const size_t n = std::min(left, buffer.size());
                    if (!input.read(buffer.data(), n)) {
                        throw std::runtime_error("short converted state read");
                    }
                    XXH64_update(&hash, buffer.data(), n);
                    left -= n;
                }
                const size_t loaded = llama_state_seq_load_file_streaming(
                    ctx_tgt, path.c_str(), id_slot, restored.data(), restored.size(),
                    &count, written, XXH64_digest(&hash));
                target_ok = loaded == written && count == tokens.size() && restored == tokens &&
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id_slot) == state.pos_tgt;
            }
        } catch (const std::exception &) {
            target_ok = false;
        }
    } else {
        target_ok = llama_state_seq_set_data_ext(ctx_tgt, state.data.main.data(), state.data.main.size(), id_slot,
                LLAMA_STATE_SEQ_FLAGS_NONE) == state.data.main.size() &&
            llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id_slot) == state.pos_tgt;
    }
    // A converted load is target-only: the q4 draft/carry cannot cross into the
    // KVarN draft layout, so it is dropped and bootstrap re-syncs it for MTP
    // destinations (bootstrap_only above).
    const bool draft_ok = target_ok && (bootstrap_only || !ctx_dft || !state.has_draft() ||
        (!converted_mode && llama_state_seq_set_data_ext(ctx_dft, state.data.drft.data(), state.data.drft.size(), id_slot,
            LLAMA_STATE_SEQ_FLAGS_NONE) == state.data.drft.size() &&
         llama_memory_seq_pos_max(llama_get_memory(ctx_dft), id_slot) == state.pos_dft));
    const bool carry_ok = draft_ok && (bootstrap_only || !ctx_dft || state.data.spec.empty() ||
        (!converted_mode && common_speculative_set_state(spec, id_slot, state.data.spec, state.pos_tgt)));
    if (!target_ok || !draft_ok || !carry_ok) {
        prompt_cache_clear(ctx_tgt, ctx_dft, spec, id_slot);
        prompt.clear();
        return miss(!target_ok ? "target restore failed" : !draft_ok ? "draft restore failed" : "MTP carry restore failed", true);
    }
    if (copy_prompt) { prompt = std::move(candidate); }
    last_reason = bootstrap_only ? "target-only snapshot restored; MTP bootstrap required" : "snapshot restored";
    return bootstrap_only ? server_prompt_cache_result::needs_bootstrap : server_prompt_cache_result::hit;
}

server_prompt_cache_result server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new,
        llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id id_slot) {
    last_reason = "no better matching snapshot";
    if (tokens_new.empty()) { return server_prompt_cache_result::unchanged; }
    const auto info = llama_model_mtp_weights_get_info(llama_get_model(ctx_tgt));
    auto best = states.end();
    int best_lcp = prompt.tokens.get_common_prefix(tokens_new);
    float best_keep = prompt.tokens.empty() ? -1.0f : float(best_lcp) / prompt.tokens.size();
    bool best_complete = ctx_dft && !prompt.tokens.empty();
    const std::string target_layout = common_prompt_cache_layout(ctx_tgt);
    const std::string draft_layout = ctx_dft ? common_prompt_cache_layout(ctx_dft) : "";
    int rejected_lcp = -1;
    std::string rejected_reason;
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp = it->prompt.tokens.get_common_prefix(tokens_new);
        if (!lcp) { continue; }
        auto reject = [&](const char * reason) {
            if (lcp > rejected_lcp) { rejected_lcp = lcp; rejected_reason = reason; }
        };
        if (it->quarantined) {
            reject("snapshot quarantined after failed restore");
            continue;
        }
        if (it->model != llama_get_model(ctx_tgt) || it->model_instance != info.model_instance) {
            reject("snapshot model differs");
            continue;
        }
        const bool convertible_candidate =
            !common_prompt_cache_layout_reusable(it->layout_tgt, target_layout) &&
            common_prompt_cache_layout_convertible(it->layout_tgt, target_layout);
        if ((!common_prompt_cache_layout_reusable(it->layout_tgt, target_layout) && !convertible_candidate) ||
                (!convertible_candidate && ctx_dft && it->has_draft() &&
                 !common_prompt_cache_layout_reusable(it->layout_dft, draft_layout))) {
            reject("snapshot model or attention layout differs");
            continue;
        }
        if (it->pos_tgt >= (llama_pos) llama_n_ctx_seq(ctx_tgt)) {
            reject("snapshot does not fit destination context");
            continue;
        }
        if (lcp < 0.25*it->prompt.tokens.size()) { continue; }
        // A convertible snapshot can only be restored whole (no partial q4->KVarN
        // resume), so require the entire snapshot to be a verified prefix.
        if (convertible_candidate && lcp != (int) it->prompt.tokens.size()) {
            continue;
        }
        if (!convertible_candidate && info.managed && (lcp < it->prompt.n_tokens() || lcp == (int) tokens_new.size())) {
            const bool can_rewind = std::any_of(it->prompt.checkpoints.begin(), it->prompt.checkpoints.end(),
                [&](const auto & c) {
                    return c->host_only() && c->n_tokens > 0 && c->n_tokens <= lcp &&
                        c->n_tokens < (int64_t) tokens_new.size() && c->pos_max < lcp;
                });
            if (!can_rewind) {
                reject("matching recurrent prefix has no usable checkpoint");
                continue;
            }
        }
        const bool complete = ctx_dft && it->has_draft() && !it->data.spec.empty();
        const float keep = float(lcp) / it->prompt.tokens.size();
        if (!info.managed) {
            if (keep > best_keep && lcp > best_lcp) {
                best = it;
                best_lcp = lcp;
                best_keep = keep;
            }
            continue;
        }
        // Reuse the longest compatible prefix first; complete MTP state only breaks
        // ties so a short speculative entry cannot hide a much longer bootstrap hit.
        if (lcp > best_lcp || (lcp == best_lcp && complete && !best_complete)) {
            best = it;
            best_lcp = lcp;
            best_complete = complete;
        }
    }
    if (best == states.end()) {
        if (rejected_lcp > best_lcp) { last_reason = rejected_reason; }
        return server_prompt_cache_result::unchanged;
    }
    auto result = restore_impl(prompt, *best, ctx_tgt, ctx_dft, spec, id_slot, info.managed, false);
    if (result == server_prompt_cache_result::needs_bootstrap) {
        result = restore_impl(prompt, *best, ctx_tgt, ctx_dft, spec, id_slot, info.managed, true);
    }
    if (result == server_prompt_cache_result::miss && restore_invalid) { best->quarantined = true; }
    if (result == server_prompt_cache_result::hit) {
        if (info.managed) { states.splice(states.end(), states, best); }
        else {
            prompt = std::move(best->prompt);
            states.erase(best);
        }
    }
    return result;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && accounted_size() > limit_size) {
            SRV_WRN(" - cache accounted-payload limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().accounted_size() / (1024.0 * 1024.0));

            states.pop_front();
            ++evictions;
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(accounted_size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().accounted_size() / (1024.0 * 1024.0));

            states.pop_front();
            ++evictions;
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), accounted_size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.accounted_size() / (1024.0 * 1024.0));
    }
}
