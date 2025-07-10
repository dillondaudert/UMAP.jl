# generate the documentation

using Documenter, UMAP, NearestNeighborDescent

makedocs(
    #modules=[UMAP],
    sitename="UMAP.jl Documentation",
    authors="Dillon Daudert",
    pages=[
        "Home" => "index.md",
        "Basic Usage" => "basic.md",
        "Advanced Usage" => "advanced.md",
        "Reference" => [
            "Public" => "ref/public.md",
            "Internal" => "ref/internal.md",
        ],
    ]
)

deploydocs(
    repo="github.com/dillondaudert/UMAP.jl.git",
    devbranch="v0.2-dev"
)