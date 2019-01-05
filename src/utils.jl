
function make_epochs_per_sample(weights::AbstractSparseMatrix, n_epochs::Integer)
    result = n_epochs .* copy(weights) ./ maximum(weights)
    nonzeros(result) .= n_epochs .* inv.(nonzeros(result)) 
    return result
end