using CairoMakie
using AbstractPlotting
using AbstractPlotting.MakieLayout
using AbstractPlotting: px


function hide_decorations!(ax)
    ax.xticksvisible = false
    ax.yticksvisible = false
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false
    ax.bottomspinevisible = false
    ax.leftspinevisible = false
    ax.topspinevisible = false
    ax.rightspinevisible = false
    ax.xgridvisible = false
    ax.ygridvisible = false
end

function plot_umap(embedding, color)
    scene, layout = layoutscene()

    layout[1, 1] = umap_ax = LAxis(scene)

    umap_ax.xautolimitmargin[] = (2 * umap_ax.xautolimitmargin[][1],
                                  2 * umap_ax.xautolimitmargin[][2])
    umap_ax.yautolimitmargin[] = (2 * umap_ax.yautolimitmargin[][1],
                                  2 * umap_ax.yautolimitmargin[][2])

    hide_decorations!(umap_ax)

    plt = scatter!(umap_ax, Point2f0.(eachcol(embedding)); color=color, markersize=5px, strokecolor = :transparent)

    cbar = layout[:, 2] = LColorbar(scene, plt, label="Number")
    cbar.width = Fixed(30)
    cbar.height = Relative(2 / 3)

    scene
end


function plot_umap_comparison((e1, c1), (e2, c2); titles, title=nothing)
    scene, layout = layoutscene()

    layout[1, 1] = ax1 = LAxis(scene, title=titles[1])
    layout[1, 2] = ax2 = LAxis(scene, title=titles[2])

    for umap_ax in (ax1, ax2)
        umap_ax.xautolimitmargin[] = (2 * umap_ax.xautolimitmargin[][1],
                                      2 * umap_ax.xautolimitmargin[][2])
        umap_ax.yautolimitmargin[] = (2 * umap_ax.yautolimitmargin[][1],
                                      2 * umap_ax.yautolimitmargin[][2])
        hide_decorations!(umap_ax)
    end

    plt1 = scatter!(ax1, Point2f0.(eachcol(e1)); color=c1, markersize=1px, strokecolor = :transparent)
    plt2 = scatter!(ax2, Point2f0.(eachcol(e2)); color=c2, markersize=1px, strokecolor = :transparent)

    cbar = layout[:, 3] = LColorbar(scene, plt1, label="Number")
    cbar.width = Fixed(30)
    cbar.height = Relative(2 / 3)

    if title !== nothing
        layout[0, :] = LText(scene, title, textsize=20, font="Noto Sans Bold",
                             color=(:black, 0.25))
    end

    scene
end
