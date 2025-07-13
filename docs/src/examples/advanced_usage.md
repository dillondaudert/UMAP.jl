```@raw html
<style>
    #documenter-page table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    #documenter-page pre, #documenter-page div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "c4d49136f1e54b9530d66f07015df7a33e49468bae7004a1e1c57b4ab13831d9"
    julia_version = "1.11.5"
-->

<pre class='language-julia'><code class='language-julia'>import Pkg</code></pre>


<pre class='language-julia'><code class='language-julia'>Pkg.activate(@__DIR__)</code></pre>


<pre class='language-julia'><code class='language-julia'>using UMAP</code></pre>


<pre class='language-julia'><code class='language-julia'>using Distances</code></pre>


<pre class='language-julia'><code class='language-julia'>using StringDistances</code></pre>

<div class="markdown"><h1 id="Advanced-Usage">Advanced Usage</h1></div>


<div class="markdown"><h2 id="Algorithm">Algorithm</h2><p>At a high level, the UMAP algorithm proceeds in the following steps:</p><pre><code class="language-julia">knns_dists = knn_search(data, knn_params)
fuzzy_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_view_params)
umap_graph = coalesce_views(fuzzy_sets, src_global_params)
embedding = initialize_embedding(umap_graph, tgt_params)
optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)</code></pre></div>


<div class="markdown"><h2 id="KNN-Search">KNN Search</h2><p>In a typical workflow, the first step of the UMAP algorithm is to find a (approximate) k-nearest neighbor graph. </p></div>


<div class="markdown"><h3 id="Example:-Approximate-neighbors-for-vector-data">Example: Approximate neighbors for vector data</h3><p>A very simple example of this is to find 4 approximate nearest neighbors for vectors in R^n using the Euclidean metric:</p></div>



<pre class='language-julia'><code class='language-julia'>xs = [rand(10) for _ in 1:10];</code></pre>

<pre class="code-output documenter-example-output" id="var-knn_params">UMAP.DescentNeighbors{Euclidean, @NamedTuple{}}(4, Euclidean(0.0), NamedTuple())</pre>
<pre class='language-julia'><code class='language-julia'>knn_params = UMAP.DescentNeighbors(4, Euclidean())</code></pre>

<pre class="code-output documenter-example-output" id="var-hash109134">([9 7 … 5 8; 7 10 … 1 4; 8 8 … 6 5; 10 5 … 4 6], [1.1200776536565635 1.1368233483216037 … 1.0165537977909198 0.660223717823423; 1.1834268667036132 1.4106504963381217 … 1.1200776536565635 1.1817726255673575; 1.2595112705830374 1.4142932138698792 … 1.4187425820388506 1.2452692059260924; 1.4645080742158803 1.4818161899364837 … 1.4189676983711252 1.2967339403276879])</pre>
<pre class='language-julia'><code class='language-julia'>UMAP.knn_search(xs, knn_params)</code></pre>

<div class="markdown"><p>The return result in this case is a tuple of 4x10 (<code>n_neighbors</code> x <code>n_points</code>) matrices, one for the indices of the nearest neighbors and the second for the distances.</p><p>e.g. <code>knn_search(xs, knn_params) -&gt; indices, distances</code></p></div>


<div class="markdown"><p>The knn parameter struct <code>DescentNeighbors</code> uses <code>NearestNeighborDescent.jl</code> to find the approximate knns of the data. It also allows passing keyword arguments to <code>nndescent</code>:</p></div>



<pre class='language-julia'><code class='language-julia'>knn_params_kw = UMAP.DescentNeighbors(4, Euclidean(), (max_iters=15,));</code></pre>

<pre class="code-output documenter-example-output" id="var-hash477790">([9 7 … 5 8; 7 10 … 1 4; 8 8 … 6 5; 10 5 … 4 6], [1.1200776536565635 1.1368233483216037 … 1.0165537977909198 0.660223717823423; 1.1834268667036132 1.4106504963381217 … 1.1200776536565635 1.1817726255673575; 1.2595112705830374 1.4142932138698792 … 1.4187425820388506 1.2452692059260924; 1.4645080742158803 1.4818161899364837 … 1.4189676983711252 1.2967339403276879])</pre>
<pre class='language-julia'><code class='language-julia'>UMAP.knn_search(xs, knn_params_kw)</code></pre>

<div class="markdown"><h3 id="Example:-Precomputed-distances">Example: Precomputed distances</h3><p>Alternatively, a precomputed distance matrix can be passed in if the pairwise distances are already known. This is done by using the <code>PrecomputedNeighbors</code> knn parameter struct (note that <code>n_neighbors</code> is still required in order to later construct the fuzzy simplicial set, and for transforming new data):</p></div>



<pre class='language-julia'><code class='language-julia'>distances = [0. 2 1;
             2 0 3;
             1 3 0];</code></pre>

<pre class="code-output documenter-example-output" id="var-knn_params_pre">UMAP.PrecomputedNeighbors{Matrix{Float64}}(2, [0.0 2.0 1.0; 2.0 0.0 3.0; 1.0 3.0 0.0])</pre>
<pre class='language-julia'><code class='language-julia'>knn_params_pre = UMAP.PrecomputedNeighbors(2, distances) </code></pre>

<pre class="code-output documenter-example-output" id="var-hash108329">([3 1 1; 2 3 2], [1.0 2.0 1.0; 2.0 3.0 3.0])</pre>
<pre class='language-julia'><code class='language-julia'>UMAP.knn_search(nothing, knn_params_pre)</code></pre>

<div class="markdown"><h3 id="Example:-Multiple-views">Example: Multiple views</h3><p>One key feature of UMAP is combining multiple, heterogeneous views of the same dataset. For the knn search step, this is set up by passing a named tuple of data views and a corresponding named tuple of knn parameter structs. The <code>knn_search</code> function then broadcasts for each (data, knn_param) pair and returns a named tuple of (indices, distances) that similarly corresponds to the input.</p><p>For example, in addition to the vector data <code>xs</code> we might also have string data:</p></div>



<pre class='language-julia'><code class='language-julia'>xs_str = [join(rand('A':'Z', 10), "") for _ in 1:10];</code></pre>


<pre class='language-julia'><code class='language-julia'>knn_params_str = UMAP.DescentNeighbors(4, RatcliffObershelp());</code></pre>

<pre class="code-output documenter-example-output" id="var-data_views">(view_1 = [[0.10284279955306941, 0.0295942483491084, 0.8387649352351909, 0.39623892420121254, 0.18628082656577694, 0.0807356489931862, 0.584269010373879, 0.7975275913366385, 0.3136172961801479, 0.4571771671449504], [0.07233610937524171, 0.24152222450737637, 0.044666437516960755, 0.06786983012023895, 0.5516536855745832, 0.9183671151818699, 0.037904704476037665, 0.3785450556474107, 0.7165629242936741, 0.9424030989711804], [0.9090119033662081, 0.5524835686404437, 0.13501969571608308, 0.5658582458007281, 0.5983238959357674, 0.964442118575702, 0.9835898859990029, 0.961741398123816, 0.15623083118926662, 0.017130571516153936], [0.9428807793975751, 0.7201475097833473, 0.9464307862557572, 0.5050263809157581, 0.8229019692796715, 0.49836552543906776, 0.581480808142305, 0.06196263598289398, 0.48403641165112976, 0.6720585098756685], [0.26665316172460274, 0.610707907824581, 0.5978522254635763, 0.689529136818924, 0.8077788612960898, 0.8736072841937906, 0.8654382119555372, 0.027098735401883678, 0.8449091027954128, 0.2814608065532753], [0.34139055112944106, 0.9528225107456801, 0.0522756136562027, 0.857023790620638, 0.07557475986190443, 0.5316999127821945, 0.15147155606924245, 0.6587711756985786, 0.9601468610856877, 0.02812922715609012], [0.41044937214337485, 0.2445865806979559, 0.13213084199704173, 0.1964924865971066, 0.7577943475024012, 0.5122315502273267, 0.5109132911184272, 0.4506813386710463, 0.02748554911966239, 0.4498230732121594], [0.4007306588816665, 0.7600194270927548, 0.5391609867739149, 0.5855316440075746, 0.9975686021040736, 0.1350333037143835, 0.3904279190086407, 0.5561733385216185, 0.40580719529305065, 0.727463961021839], [0.05397457585696286, 0.5625212817015149, 0.9454696508870256, 0.4136468396239823, 0.25001919171609555, 0.33756963690702524, 0.9462512041551828, 0.37020949087944166, 0.9492205625465947, 0.04628054286391792], [0.3573441364573897, 0.6594053958425605, 0.4095312893447579, 0.756683411509623, 0.9635192494282159, 0.027634399552627587, 0.1049364659406431, 0.33544020674026, 0.8659222930177561, 0.5752643336847262]], view_2 = ["MAPBBBCRPN", "ZIYODMPPSF", "IYDYLNWWCT", "QVBQLXMHPD", "YBIFLBHHEA", "JWNFJYZLTY", "KWIPMBSDRM", "HLPWBIWGOO", "EHSIWZYBEP", "TOITCZKDEX"])</pre>
<pre class='language-julia'><code class='language-julia'>data_views = (view_1=xs, 
              view_2=xs_str)</code></pre>

<pre class="code-output documenter-example-output" id="var-knn_params_views">(view_1 = UMAP.DescentNeighbors{Euclidean, @NamedTuple{}}(4, Euclidean(0.0), NamedTuple()), view_2 = UMAP.DescentNeighbors{RatcliffObershelp, @NamedTuple{}}(4, RatcliffObershelp(), NamedTuple()))</pre>
<pre class='language-julia'><code class='language-julia'>knn_params_views = (view_1=knn_params, 
                    view_2=knn_params_str)</code></pre>

<pre class="code-output documenter-example-output" id="var-hash157805">(view_1 = ([9 7 … 5 8; 7 10 … 1 4; 8 8 … 6 5; 10 5 … 4 6], [1.1200776536565635 1.1368233483216037 … 1.0165537977909198 0.660223717823423; 1.1834268667036132 1.4106504963381217 … 1.1200776536565635 1.1817726255673575; 1.2595112705830374 1.4142932138698792 … 1.4187425820388506 1.2452692059260924; 1.4645080742158803 1.4818161899364837 … 1.4189676983711252 1.2967339403276879]), view_2 = ([7 1 … 2 9; 2 3 … 6 3; 8 9 … 5 5; 5 7 … 8 7], [0.7 0.7 … 0.7 0.7; 0.7 0.7 … 0.7 0.8; 0.8 0.7 … 0.7 0.8; 0.8 0.7 … 0.7 0.8]))</pre>
<pre class='language-julia'><code class='language-julia'>UMAP.knn_search(data_views, knn_params_views)</code></pre>

<div class="markdown"><h2 id="Fuzzy-Simplicial-Sets">Fuzzy Simplicial Sets</h2><p>Once we have one or more set of knns for our data (one for each view), we can construct a global fuzzy simplicial set. This is done via the function</p><p><code>fuzzy_simplicial_set(...) -&gt; umap_graph::SparseMatrixCSC</code></p><p>A global fuzzy simplicial set is constructed <strong>for each view</strong> of the data with construction paramaterized by the <code>SourceViewParams</code> struct. If there is more than one view, their results are combined to return a single fuzzy simplicial set (represented as a weighted, undirected graph).</p></div>


<div class="markdown"><h3 id="Example:-Fuzzy-simplicial-set---one-view">Example: Fuzzy simplicial set - one view</h3><p>To create a fuzzy simplicial set for our original dataset of vectors:</p></div>


<pre class="code-output documenter-example-output" id="var-src_view_params">UMAP.SourceViewParams{Int64}(1, 1, 1)</pre>
<pre class='language-julia'><code class='language-julia'>src_view_params = UMAP.SourceViewParams(1, 1, 1)</code></pre>

<pre class="code-output documenter-example-output" id="var-knns_dists">([9 7 … 5 8; 7 10 … 1 4; 8 8 … 6 5; 10 5 … 4 6], [1.1200776536565635 1.1368233483216037 … 1.0165537977909198 0.660223717823423; 1.1834268667036132 1.4106504963381217 … 1.1200776536565635 1.1817726255673575; 1.2595112705830374 1.4142932138698792 … 1.4187425820388506 1.2452692059260924; 1.4645080742158803 1.4818161899364837 … 1.4189676983711252 1.2967339403276879])</pre>
<pre class='language-julia'><code class='language-julia'>knns_dists = UMAP.knn_search(xs, knn_params)</code></pre>

<pre class="code-output documenter-example-output" id="var-hash110181">10×10 SparseArrays.SparseMatrixCSC{Float64, Int64} with 52 stored entries:
  ⋅          ⋅         ⋅         ⋅        …  0.720641  0.330453  1.0       0.0648782
  ⋅          ⋅         ⋅         ⋅           1.0       0.358107   ⋅        0.362967
  ⋅          ⋅         ⋅         ⋅           1.0       0.306533   ⋅         ⋅ 
  ⋅          ⋅         ⋅         ⋅           0.150732  1.0       0.178812  0.55336
  ⋅         0.27892   0.408287  0.860412      ⋅        0.328473  1.0       0.432832
  ⋅          ⋅        0.285171   ⋅        …   ⋅        0.244896  0.518186  1.0
 0.720641   1.0       1.0       0.150732      ⋅        1.0        ⋅         ⋅ 
 0.330453   0.358107  0.306533  1.0          1.0        ⋅         ⋅        1.0
 1.0         ⋅         ⋅        0.178812      ⋅         ⋅         ⋅         ⋅ 
 0.0648782  0.362967   ⋅        0.55336       ⋅        1.0        ⋅         ⋅ </pre>
<pre class='language-julia'><code class='language-julia'>UMAP.fuzzy_simplicial_set(knns_dists, knn_params, src_view_params)</code></pre>

<div class="markdown"><h3 id="Example:-Fuzzy-simplicial-set---multiple-views">Example: Fuzzy simplicial set - multiple views</h3><p>As before, multiple views can be passed to <code>fuzzy_simplicial_set</code> - each parameterized by its own <code>SourceViewParams</code> - and combined into a single, global fuzzy simplicial set.</p><p>Using our combination of vector and string data:</p></div>


<pre class="code-output documenter-example-output" id="var-knns_dists_views">(view_1 = ([9 7 … 5 8; 7 10 … 1 4; 8 8 … 6 5; 10 5 … 4 6], [1.1200776536565635 1.1368233483216037 … 1.0165537977909198 0.660223717823423; 1.1834268667036132 1.4106504963381217 … 1.1200776536565635 1.1817726255673575; 1.2595112705830374 1.4142932138698792 … 1.4187425820388506 1.2452692059260924; 1.4645080742158803 1.4818161899364837 … 1.4189676983711252 1.2967339403276879]), view_2 = ([2 7 … 8 9; 7 1 … 6 3; 8 3 … 5 5; 4 9 … 2 7], [0.7 0.7 … 0.7 0.7; 0.7 0.7 … 0.7 0.8; 0.8 0.7 … 0.7 0.8; 0.8 0.7 … 0.7 0.8]))</pre>
<pre class='language-julia'><code class='language-julia'>knns_dists_views = UMAP.knn_search(data_views, knn_params_views)</code></pre>

<pre class="code-output documenter-example-output" id="var-src_view_params_views">(view_1 = UMAP.SourceViewParams{Int64}(1, 1, 1), view_2 = UMAP.SourceViewParams{Int64}(1, 1, 1))</pre>
<pre class='language-julia'><code class='language-julia'>src_view_params_views = (view_1=src_view_params, 
                         view_2=src_view_params)</code></pre>

<pre class="code-output documenter-example-output" id="var-fsset_views">(view_1 = sparse([7, 8, 9, 10, 5, 7, 8, 10, 5, 6  …  1, 4, 5, 6, 1, 2, 4, 5, 6, 8], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3  …  9, 9, 9, 9, 10, 10, 10, 10, 10, 10], [0.7206410878673053, 0.33045284489307875, 1.0, 0.06487824618889006, 0.27892033480408185, 1.0, 0.3581066352292656, 0.36296725188522677, 0.40828728639604356, 0.28517102556086865  …  1.0, 0.17881190697812666, 1.0, 0.5181862353431707, 0.06487824618889006, 0.36296725188522677, 0.5533599488120708, 0.43283234847631036, 0.9999999999999999, 1.0], 10, 10), view_2 = sparse([2, 4, 5, 7, 8, 1, 3, 4, 6, 7  …  2, 4, 5, 6, 8, 10, 3, 5, 7, 9], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  9, 9, 9, 9, 9, 9, 10, 10, 10, 10], [1.0, 2.7607725720371694e-6, 2.7607725720371694e-6, 1.0, 0.33333708943163637, 1.0, 1.0, 0.3333352489218768, 2.7607725720371694e-6, 1.0  …  1.0, 0.3333352489218768, 1.0, 1.0, 1.0, 1.0, 0.3333352489218768, 0.3333352489218768, 0.3333352489218768, 1.0], 10, 10))</pre>
<pre class='language-julia'><code class='language-julia'>fsset_views = UMAP.fuzzy_simplicial_set(knns_dists_views, knn_params_views, src_view_params_views)</code></pre>

<div class="markdown"><h3 id="Example:-Combining-views&#39;-fuzzy-simplicial-sets">Example: Combining views' fuzzy simplicial sets</h3><p>We need a single umap graph (i.e. global fuzzy simplicial set) in order to perform optimization, so if there are multiple dataset views we must combine their sets.</p><p>The views' fuzzy sets are combined left-to-right according to <code>mix_ratio</code>:</p></div>


<pre class="code-output documenter-example-output" id="var-src_gbl_params">UMAP.SourceGlobalParams{Float64}(0.5)</pre>
<pre class='language-julia'><code class='language-julia'>src_gbl_params = UMAP.SourceGlobalParams(0.5)</code></pre>

<pre class="code-output documenter-example-output" id="var-_graph">10×10 SparseArrays.SparseMatrixCSC{Float64, Int64} with 74 stored entries:
  ⋅        0.920191   ⋅        0.441015  …  1.0       0.975357  0.399881  0.375719
 0.920191   ⋅        0.9129    0.837453     1.0       0.208022  0.803357  0.262165
  ⋅        0.9129     ⋅         ⋅           0.30904   1.0        ⋅        0.894578
 0.441015  0.837453   ⋅         ⋅           0.900654  0.344788  0.895274  0.371694
 0.427113  0.318419  0.418015  1.0           ⋅        0.285834  1.0       1.0
  ⋅        0.503781  0.999743   ⋅        …   ⋅        0.518293  1.0       0.596734
 1.0       1.0       0.30904   0.900654      ⋅        1.0        ⋅        0.821386
 0.975357  0.208022  1.0       0.344788     1.0        ⋅        0.818766  0.257176
 0.399881  0.803357   ⋅        0.895274      ⋅        0.818766   ⋅        0.894108
 0.375719  0.262165  0.894578  0.371694     0.821386  0.257176  0.894108   ⋅ </pre>
<pre class='language-julia'><code class='language-julia'>_graph = UMAP.coalesce_views(fsset_views, src_gbl_params)</code></pre>

<div class="markdown"><h2 id="Initialize-and-optimize-target-embedding">Initialize and optimize target embedding</h2><ul><li><p>initialize target space membership function and gradient functions</p></li><li><p>initialize target space embedding</p></li><li><p>optimize target embedding</p></li></ul></div>


<div class="markdown"><h2 id="Initialize-target-embedding">Initialize target embedding</h2><p>The target space and initialization method can be parameterized by the <code>TargetParams</code> struct:</p><pre><code class="language-julia">struct TargetParams{M, D, I, F}
	manifold::M
	metric::D
	init::I
	memb_params::F
end</code></pre><p>It is possible to specify the target manifold, a distance metric in the target space <code>metric</code>, and an initialization method. </p><p>The default target space is d-dimensional Euclidean space, with the squared Euclidean distance metric. Two initialization methods are provided: random and spectral layout.</p></div>


<div class="markdown"><h3 id="Example:-Initializing-vectors-in-R^2">Example: Initializing vectors in R^2</h3></div>


<pre class="code-output documenter-example-output" id="var-tgt_params">UMAP.TargetParams{UMAP._EuclideanManifold{2}, SqEuclidean, UMAP.UniformInitialization, Nothing}(UMAP._EuclideanManifold{2}(), SqEuclidean(0.0), UMAP.UniformInitialization(), nothing)</pre>
<pre class='language-julia'><code class='language-julia'>tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold{2}(), SqEuclidean(), UMAP.UniformInitialization(), nothing)</code></pre>


<pre class='language-julia'><code class='language-julia'>umap_graph = UMAP.fuzzy_simplicial_set(knns_dists, knn_params, src_view_params);</code></pre>

<pre class="code-output documenter-example-output" id="var-xs_embed">10-element Vector{Vector{Float64}}:
 [-4.4090401398747385, 5.383709597291055]
 [2.929515493849429, 8.95391226848162]
 [6.183735414656017, -0.06836711594628042]
 [4.781798597955833, -4.432207338115369]
 [4.435016898846598, -2.054561783632667]
 [-7.292946000948115, -3.6828558310221453]
 [-6.158330468751556, -6.742464490916104]
 [-0.3697772630964593, 5.820527585120207]
 [-9.694062393210805, -7.458190885007914]
 [4.441380475860509, 1.9961639792313495]</pre>
<pre class='language-julia'><code class='language-julia'>xs_embed = UMAP.initialize_embedding(umap_graph, tgt_params)</code></pre>

<div class="markdown"><h3 id="MembershipFnParams">MembershipFnParams</h3><p>These parameters control the layout of points embedded in the target space by adjusting the membership function. <em>TO DO</em>.</p><pre><code class="language-julia">struct MembershipFnParams
	min_dist
	spread
	a
	b
end</code></pre></div>


<pre class="code-output documenter-example-output" id="var-a">(0.11497598978623988, 1.929235340778851)</pre>
<pre class='language-julia'><code class='language-julia'>a, b = UMAP.fit_ab(1, 1)</code></pre>

<pre class="code-output documenter-example-output" id="var-full_tgt_params">UMAP.TargetParams{UMAP._EuclideanManifold{2}, SqEuclidean, UMAP.UniformInitialization, UMAP.MembershipFnParams{Float64}}(UMAP._EuclideanManifold{2}(), SqEuclidean(0.0), UMAP.UniformInitialization(), UMAP.MembershipFnParams{Float64}(1.0, 1.0, 0.11497598978623988, 1.929235340778851))</pre>
<pre class='language-julia'><code class='language-julia'>full_tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold{2}(), SqEuclidean(), UMAP.UniformInitialization(), UMAP.MembershipFnParams(1., 1., a, b))</code></pre>

<div class="markdown"><h2 id="Optimize-target-embedding">Optimize target embedding</h2><p>The embedding is optimized by minimizing the fuzzy set cross entropy loss between the  two fuzzy set representations of the data. </p></div>


<div class="markdown"><h3 id="Example:-Optimize-one-epoch">Example: Optimize one epoch</h3><p>The optimization process is parameterized by the struct <code>OptimizationParams</code>:</p><pre><code class="language-julia">struct OptimizationParams
	n_epochs           # number of epochs to perform optimization
	lr                 # learning rate
    repulsion_strength # weight to give negative samples
    neg_sample_rate    # number of negative samples per positive sample
end</code></pre></div>


<pre class="code-output documenter-example-output" id="var-opt_params">UMAP.OptimizationParams(1, 1.0, 1.0, 5)</pre>
<pre class='language-julia'><code class='language-julia'>opt_params = UMAP.OptimizationParams(1, 1., 1., 5)</code></pre>

<pre class="code-output documenter-example-output" id="var-hash924224">10-element Vector{Vector{Float64}}:
 [-4.6544432223353684, 4.514695900568833]
 [2.7887316348780415, 7.984910924947306]
 [5.277907703406434, -1.5048716389945045]
 [0.8706322595247891, -1.6913813142865548]
 [2.6362354626729654, -4.476176166419125]
 [-6.932898665363556, -3.8431681915568885]
 [-4.560827539985891, -5.278672344324341]
 [0.7384093035362213, 3.2404796299574494]
 [-8.644828173520267, -6.386159304780086]
 [2.263613674330533, 2.0934051845703348]</pre>
<pre class='language-julia'><code class='language-julia'>UMAP.optimize_embedding!(xs_embed, umap_graph, full_tgt_params, opt_params)</code></pre>

<!-- PlutoStaticHTML.End -->
```

