function load_fomc_classifier(; kwrgs...)
    model_name = "gtfintechlab/FOMC-RoBERTa"
    
    tkr = Transformers.load_tokenizer(model_name)
    cfg = Transformers.HuggingFace.HGFConfig(Transformers.load_config(model_name); kwrgs...)
    mod = Transformers.load_model(model_name, "ForSequenceClassification"; config = cfg)
    
    return tkr, mod
end

function load_fomc_cmlm(; kwrgs...)
    model_name = "karoldobiczek/relitc-FOMC-CMLM"
    
    tkr = Transformers.load_tokenizer(model_name)
    cfg = Transformers.HuggingFace.HGFConfig(Transformers.load_config(model_name); kwrgs...)
    mod = Transformers.load_model(model_name, "ForMaskedLM"; config = cfg)

    return tkr, mod
end
