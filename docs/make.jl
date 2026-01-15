# generate the documentation

using Documenter, UMAP, NearestNeighborDescent
using PlutoStaticHTML

# run Pluto notebooks to generate files for Documenter
notebook_dir = joinpath(@__DIR__, "src", "examples")
notebook_files = ["mnist.jl", "fashion_mnist.jl", "advanced_usage.jl", "composite.jl"]
build_opts = PlutoStaticHTML.BuildOptions(
    notebook_dir;
    output_format=documenter_output,
    previous_dir=notebook_dir,
)
output_opts = PlutoStaticHTML.OutputOptions(;
    show_output_above_code=true,
)

build_notebooks(
    build_opts,
    notebook_files,
    output_opts
)

makedocs(
    modules=[UMAP],
    sitename="UMAP.jl Documentation",
    authors="Dillon Daudert",
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "MNIST" => "examples/mnist.md",
            "Fashion MNIST" => "examples/fashion_mnist.md"
        ],
        "Usage" => [
            "Advanced Usage" => "examples/advanced_usage.md",
            "Composite Views" => "examples/composite.md",
        ],
        "Loss Function" => "loss.md",
        "Reference" => [
            "Public API" => "ref/public.md",
            "Internal" => "ref/internal.md",
        ],
    ],
    warnonly=:doctest,
    format=Documenter.HTML(
        size_threshold=nothing, # disable size threshold for HTML output
    )
)

deploydocs(
    repo="github.com/dillondaudert/UMAP.jl.git",
    devbranch="master",
    versions=["stable" => "v^", "dev" => "dev"]
)