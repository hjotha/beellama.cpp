#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // checkpoint count cannot wrap through size_t during snapshot budgeting
    argv = {"binary_name", "--ctx-checkpoints", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    {
        common_params adaptive;
        argv = {
            "binary_name", "--ctx-size", "1000", "--ctx-size-mtp", "600",
            "--mtp-max-tokens", "500", "--fit", "off", "--parallel", "1",
            "--spec-type", "draft-mtp",
        };
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), adaptive, LLAMA_EXAMPLE_SERVER));
        assert(adaptive.ctx_size_mtp == 600);
        assert(adaptive.mtp_max_tokens == 500);
        assert(adaptive.split_mtp_weights);
        assert(common_context_is_adaptive(adaptive));
        assert(common_context_mtp_limit(adaptive) == 500);
        assert(common_context_adaptive_error(adaptive).empty());

        const auto budget = common_context_budget_for_task(adaptive, 499, -1, true);
        assert(budget.prompt_tokens == 499);
        assert(budget.output_reserve == 4096);
        assert(budget.total_tokens == 4595);
        assert(common_context_profile_for_budget(adaptive, 499) == COMMON_CONTEXT_PROFILE_MTP);
        assert(common_context_profile_for_budget(adaptive, 500) == COMMON_CONTEXT_PROFILE_MTP);
        assert(common_context_profile_for_budget(adaptive, 501) == COMMON_CONTEXT_PROFILE_LONG);
        assert(common_context_output_reserve(adaptive, 0, true) == 0);
        assert(common_context_output_reserve(adaptive, -1, false) == 0);
        assert(common_context_output_reserve(adaptive, 7, true) == 7);
        adaptive.n_predict = 123;
        assert(common_context_output_reserve(adaptive, -1, true) == 123);

        adaptive.mtp_max_tokens = 0;
        assert(common_context_mtp_limit(adaptive) == 600);
        assert(common_context_adaptive_error(adaptive, 599) ==
               "--ctx-size-mtp must not exceed the long context size");

        common_params capped = adaptive;
        capped.mtp_max_tokens = 500;
        assert(common_context_adaptive_error(capped, 512) ==
               "--ctx-size-mtp must not exceed the long context size");
        capped.ctx_size_mtp = 400;
        capped.mtp_max_tokens = 400;
        assert(common_context_adaptive_error(capped, 512).empty());
    }

    {
        auto make_adaptive = [] {
            common_params value;
            value.n_ctx = 1000;
            value.ctx_size_mtp = 600;
            value.mtp_max_tokens = 500;
            value.n_parallel = 1;
            value.fit_params = false;
            value.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            return value;
        };
        auto expect_error = [&](const common_params & value, const char * expected) {
            assert(common_context_adaptive_error(value) == expected);
        };

        common_params disabled;
        assert(!common_context_is_adaptive(disabled));
        assert(common_context_profile_for_budget(disabled, 0) == COMMON_CONTEXT_PROFILE_LONG);
        assert(common_context_adaptive_error(disabled).empty());
        disabled.mtp_max_tokens = 1;
        expect_error(disabled, "--mtp-max-tokens requires --ctx-size-mtp");

        auto adaptive = make_adaptive();
        adaptive.split_mtp_weights = false;
        assert(common_context_adaptive_normalize(adaptive).empty());
        assert(adaptive.split_mtp_weights);
        adaptive.ctx_size_mtp = -1;
        expect_error(adaptive, "--ctx-size-mtp must be non-negative");
        adaptive = make_adaptive();
        adaptive.mtp_max_tokens = -1;
        expect_error(adaptive, "--mtp-max-tokens must be non-negative");
        adaptive = make_adaptive();
        adaptive.mtp_max_tokens = 601;
        expect_error(adaptive, "--mtp-max-tokens must satisfy 0 < limit <= --ctx-size-mtp");
        adaptive = make_adaptive();
        adaptive.n_ctx = 500;
        expect_error(adaptive, "--ctx-size-mtp must not exceed the long context size");
        adaptive = make_adaptive();
        adaptive.n_parallel = 2;
        expect_error(adaptive, "adaptive context requires --parallel 1");
        adaptive = make_adaptive();
        adaptive.fit_params = true;
        expect_error(adaptive, "adaptive context requires --fit off");
        adaptive = make_adaptive();
        adaptive.no_alloc = true;
        expect_error(adaptive, "adaptive context requires allocated model tensors (no-alloc is unsupported)");
        adaptive = make_adaptive();
        adaptive.kv_paged = true;
        expect_error(adaptive, "adaptive context requires traditional KV (paged KV is unsupported)");
        adaptive = make_adaptive();
        adaptive.kv_unified_per_slot = 1;
        expect_error(adaptive, "adaptive context does not support --kv-unified-per-slot");
        adaptive = make_adaptive();
        adaptive.devices = { nullptr };
        expect_error(adaptive, "adaptive context requires one CUDA device");
        adaptive = make_adaptive();
        adaptive.devices = { nullptr, nullptr };
        expect_error(adaptive, "adaptive context requires one CUDA device");
        assert(common_context_prepare_devices(adaptive) ==
               "adaptive context requires exactly one selected CUDA device");
        adaptive = make_adaptive();
        adaptive.n_gpu_layers = 0;
        adaptive.devices = { nullptr };
        assert(common_context_adaptive_error(adaptive).empty());
        assert(common_context_prepare_devices(adaptive).empty());
        assert(adaptive.devices.size() == 1 && adaptive.devices.front() == nullptr);
        adaptive = make_adaptive();
        adaptive.mmproj.path = "mmproj.gguf";
        expect_error(adaptive, "adaptive context does not support multimodal models");
        adaptive = make_adaptive();
        adaptive.lora_adapters.push_back({});
        expect_error(adaptive, "adaptive context does not support LoRA or control vectors");
        adaptive = make_adaptive();
        adaptive.control_vectors.push_back({});
        expect_error(adaptive, "adaptive context does not support LoRA or control vectors");
        adaptive = make_adaptive();
        adaptive.sleep_idle_seconds = 1;
        expect_error(adaptive, "adaptive context does not support automatic sleep");
        adaptive = make_adaptive();
        adaptive.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
        expect_error(adaptive, "adaptive context supports only draft-mtp speculative decoding");
        adaptive = make_adaptive();
        adaptive.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
        expect_error(adaptive, "adaptive context requires draft-mtp speculative decoding");
        adaptive = make_adaptive();
        adaptive.speculative.synth_len = 2.0;
        expect_error(adaptive, "adaptive context does not support synthetic speculative decoding");
        adaptive = make_adaptive();
        adaptive.speculative.draft.mparams.path = "draft.gguf";
        expect_error(adaptive, "adaptive context requires the resident model MTP head; external draft models are unsupported");

        try {
            common_context_budget_for_task(adaptive, -1, 0, true);
            assert(false);
        } catch (const std::invalid_argument & error) {
            assert(std::string(error.what()) == "adaptive context prompt token count must be non-negative");
        }
        try {
            common_context_budget_for_task(adaptive, std::numeric_limits<int64_t>::max(), 1, true);
            assert(false);
        } catch (const std::invalid_argument & error) {
            assert(std::string(error.what()) == "adaptive context token budget overflow");
        }

        common_params parse_adaptive;
        argv = {"binary_name", "--ctx-size-mtp", "600", "--fit", "off", "--parallel", "1",
                "--mtp-max-tokens", "601", "--spec-type", "draft-mtp"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), parse_adaptive, LLAMA_EXAMPLE_SERVER));

        argv = {"binary_name", "--ctx-size", "500", "--ctx-size-mtp", "600", "--fit", "off", "--parallel", "1", "--spec-type", "draft-mtp"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), parse_adaptive, LLAMA_EXAMPLE_SERVER));

        argv = {"binary_name", "--ctx-size-mtp", "600", "--parallel", "1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), parse_adaptive, LLAMA_EXAMPLE_SERVER));

        argv = {"binary_name", "--ctx-size-mtp", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), parse_adaptive, LLAMA_EXAMPLE_SERVER));

        argv = {"binary_name", "--mtp-max-tokens", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), parse_adaptive, LLAMA_EXAMPLE_SERVER));

        common_params parse_external_draft;
        argv = {"binary_name", "--model", "model.gguf", "--ctx-size", "1000", "--ctx-size-mtp", "600",
                "--fit", "off", "--parallel", "1", "--spec-type", "draft-mtp",
                "--model-draft", "draft.gguf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), parse_external_draft, LLAMA_EXAMPLE_SERVER));
    }

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    {
        common_params power_params;
        argv = {
            "binary_name",
            "--gpu-power-prefill", "200",
            "--gpu-power-decode", "165",
            "--gpu-power-device", "1",
        };
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), power_params, LLAMA_EXAMPLE_SERVER));
        assert(power_params.gpu_power_prefill == 200);
        assert(power_params.gpu_power_decode == 165);
        assert(power_params.gpu_power_device == 1);

        common_params mem_params;
        argv = {
            "binary_name",
            "--gpu-mem-clock-decode", "10501",
            "--gpu-mem-clock-prefill", "10251",
        };
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mem_params, LLAMA_EXAMPLE_SERVER));
        assert(mem_params.gpu_mem_clock_decode == 10501);
        assert(mem_params.gpu_mem_clock_prefill == 10251);

        common_params incomplete_power_params;
        argv = {"binary_name", "--gpu-power-prefill", "200"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), incomplete_power_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_GPU_POWER_PREFILL", "200", true);
    setenv("LLAMA_ARG_GPU_POWER_DECODE", "165", true);
    setenv("LLAMA_ARG_GPU_POWER_DEVICE", "1", true);
    common_params power_env_params;
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), power_env_params, LLAMA_EXAMPLE_SERVER));
    assert(power_env_params.gpu_power_prefill == 200);
    assert(power_env_params.gpu_power_decode == 165);
    assert(power_env_params.gpu_power_device == 1);
    unsetenv("LLAMA_ARG_GPU_POWER_PREFILL");
    unsetenv("LLAMA_ARG_GPU_POWER_DECODE");
    unsetenv("LLAMA_ARG_GPU_POWER_DEVICE");

    setenv("LLAMA_ARG_GPU_MEM_CLOCK_DECODE", "10501", true);
    setenv("LLAMA_ARG_GPU_MEM_CLOCK_PREFILL", "10251", true);
    common_params mem_env_params;
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mem_env_params, LLAMA_EXAMPLE_SERVER));
    assert(mem_env_params.gpu_mem_clock_decode == 10501);
    assert(mem_env_params.gpu_mem_clock_prefill == 10251);
    unsetenv("LLAMA_ARG_GPU_MEM_CLOCK_DECODE");
    unsetenv("LLAMA_ARG_GPU_MEM_CLOCK_PREFILL");
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
