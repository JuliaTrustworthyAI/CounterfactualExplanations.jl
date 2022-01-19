function _flat(x)
    n = length(first(x))
    reshape(collect(Iterators.flatten(x)), :, n)
end