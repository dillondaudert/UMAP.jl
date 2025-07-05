# generate the documentation

using Documenter, UMAP

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
