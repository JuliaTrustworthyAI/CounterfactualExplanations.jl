include("get_attributions.jl");

mutable struct RelitcGenerator <: AbstractNonGradientBasedGenerator
    filling_scheme::Union{Nothing, Function}
    attribution_generator::Py
    cmlm
end

function RelitcGenerator(;
    filling_scheme::Union{Nothing, Function}=uncertainty_filling,
    attribution_generator::Py=load_scorer(),
    cmlm = load_fomc_cmlm()
)
    return RelitcGenerator(filling_scheme, attribution_generator, cmlm)
end

function relitc!(ce::AbstractCounterfactualExplanation)
    text = ce.x
    attributions = get_attributions(text, scorer)
    cls_tkr = ce.M[1]
    cls_mod = ce.M[2]

    cmlm_tkr = ce.generator.cmlm[1]
    cmlm_mod = ce.generator.cmlm[2]
    
    word_attributions = group_into_words(text, attributions, cls_tkr)
    idx_to_mask = get_top_k_idx(word_attributions)

    toks = decode(cls_tkr, encode(cls_tkr, text).token)
    mask_toks_at_idx(toks, idx_to_mask)

    cmlm_decoded = decode(cmlm_tkr, encode(cmlm_tkr, masked_text).token)

    idx_to_mask = get_idx_cmlm(cmlm_decoded)

    left_to_right_filling(copy(cmlm_decoded), idx_to_mask, cmlm_mod, cmlm_tkr)
end






function group_into_words(text, attributions, cls_tkr)
    toks = decode(cls_tkr, encode(cls_tkr, text).token)
    word_attributions = []
    for (i, (dec_tok, attrib)) in enumerate(zip(toks, attributions))
        if startswith(dec_tok, "<")
            continue
        elseif length(word_attributions) == 0 || startswith(dec_tok, " ")
            push!(word_attributions, ([i], [attrib[1]], [attrib[2]]))
        else 
            last_processed = last(word_attributions)
            push!(last_processed[1], i)
            push!(last_processed[2], attrib[1])
            push!(last_processed[3], attrib[2])
        end
    end
    return word_attributions
end

function get_top_k_idx(attributions, k=10)
    sorted = sort(attributions, by = x -> -maximum(x[3]))
    idx_to_mask = []
    for row in first(sorted, k)
        append!(idx_to_mask, row[1])
    end
    return idx_to_mask
end

function mask_toks_at_idx(toks, idx_to_mask)
    masked_text = Vector{Char}()
    for (i, token) in enumerate(toks)
        if startswith(token, "<")
            continue
        elseif i in idx_to_mask
            append!(masked_text, " [MASK]")
        else
            append!(masked_text, token)
        end
    end
    
    return String(masked_text)
end

function get_idx_cmlm(cmlm_decoded)
    idx_to_mask = []
    for (i, tok) in enumerate(cmlm_decoded)
        if tok == "[MASK]"
            push!(idx_to_mask, i)
        end
    end
    return idx_to_mask
end

function merge_tokens(tokens, idx_to_mask=[])
    merged_text = Vector{Char}()
    for (i, token) in enumerate(tokens)
        if i in idx_to_mask
            append!(merged_text, " [MASK]")
        else
            append!(merged_text, " " * token)
        end
    end
    
    return chop(String(merged_text), head=1, tail=0)
end

function group_into_words(cmlm_out, delim="##")
    word_list = []
    for token in cmlm_out
        if startswith(delim, token) && length(word_list) != 0
            last(word_list) = last(word_list) * chop(token, head=2, tail=0)
        else 
            push(word_list, token)
        end
    end
    return word_list
end

function left_to_right_filling(tokens, mask_positions, model, tokenizer)
    if length(mask_positions) == 0
        return merge_tokens(tokens)
    end

    masked_text = merge_tokens(tokens, mask_positions)
    # println(masked_text)
    
    out = decode(cmlm_tkr, cmlm_model(encode(cmlm_tkr, masked_text)).logit)
    
    mask_positions = sort(mask_positions)
    next_position = popfirst!(mask_positions)

    next_token = out[next_position+1]

    tokens[next_position] = next_token

    return left_to_right_filling(tokens, mask_positions, model, tokenizer)
end

function uncertainty_filling(tokens, mask_positions, model, tokenizer)
    if length(mask_positions) == 0
        return merge_tokens(tokens)
    end

    masked_text = merge_tokens(tokens, mask_positions)
    # println(masked_text)

    logits = cmlm_model(encode(cmlm_tkr, masked_text)).logit
    out = decode(cmlm_tkr, logits)

    probs = softmax(logits[:, mask_positions, :], dims=1)
    
    entrs = []
    for i in 1:length(mask_positions)
        push!(entrs, entropy(probs[:, i]))
    end
    
    next_position = mask_positions[argmin(entrs)]
    filter!(x -> x != next_position, mask_positions)
    
    next_token = out[next_position+1]

    tokens[next_position] = next_token
    return uncertainty_filling(tokens, mask_positions, model, tokenizer)
end

function uncertainty_filling(tokens, mask_positions, model, tokenizer)
    if length(mask_positions) == 0
        return merge_tokens(tokens)
    end

    masked_text = merge_tokens(tokens, mask_positions)
    # println(masked_text)

    logits = cmlm_model(encode(cmlm_tkr, masked_text)).logit
    out = decode(cmlm_tkr, logits)

    probs = softmax(logits[:, mask_positions, :], dims=1)
    
    entrs = []
    for i in 1:length(mask_positions)
        push!(entrs, entropy(probs[:, i]))
    end
    
    next_position = mask_positions[argmin(entrs)]
    filter!(x -> x != next_position, mask_positions)
    
    next_token = out[next_position+1]

    tokens[next_position] = next_token
    return uncertainty_filling(tokens, mask_positions, model, tokenizer)
end