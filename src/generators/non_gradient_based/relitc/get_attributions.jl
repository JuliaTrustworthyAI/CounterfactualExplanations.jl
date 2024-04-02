function load_scorer()
    transformers_interpret = PythonCall.pyimport("transformers_interpret")
    transformers = PythonCall.pyimport("transformers")

    # load pre-trained classifier and corresponding tokenizer
    model = transformers.RobertaForSequenceClassification.from_pretrained("model", local_files_only=true)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa")

    scorer = transformers_interpret.SequenceClassificationExplainer(model, tokenizer, attribution_type="lig")

    return scorer
end

function get_attributions(text, scorer)
    attribs = scorer(text, index=0, internal_batch_size=1)
    attributions = pyconvert(Array{Tuple{String, Float64}}, attribs)
    return attributions
end