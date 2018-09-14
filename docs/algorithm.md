# Overview
"UMAP uses local manifold approximations and patches together their fuzzy
simplicial set representations. This constructs a topological representation
of the data.

Given a low dimensional representation of the data, a similar process can be
used to construct an equivalent topological representation.

UMAP then optimizes the layout of the data representation in the low dimensional
space, minimizing the cross-entropy between the two topological
representations."

1. Approximate a manifold on which the data is assumed to lie
2. Construct a fuzzy simplicial set representation of the approximated manifold
3. Construct another fuzzy simplicial set representation of a low dimensional
manifold, `R^d`
4. Optimize the representation

## Estimating the manifold
Assuming the data points **x** in *X* are uniformly distributed on a manifold
*M* with respect to the Riemannian metric *g*, then a ball centered at **x_i**
that contains exactly the *k*-nearest neighbors of **x_i** should have a fixed
volume regardless of the choice of **x_i**.

From Lemma 1 of the paper, the geodesic distance of **x_i** to its neighbors
can be approximated by normalizing distances with respect to the distance to the
*k*-th nearest neighbor of **x_i**.

This results in a family of discrete metric spaces (one for each **x_i**) that
can be merged into a consistent global structure by converting the metric spaces
into fuzzy simplicial sets.

## Constructing fuzzy simplicial sets
"Each metric space in the family can be translated into a fuzzy simplicial set
via the fuzzy singular set functor, distilling the topological information
while still retaining metric information in the fuzzy structure. Ironing out
the incompatibilities of the resulting family of fuzzy simplicial sets can be
done by simply taking a (fuzzy) union across the entire family. The result is a
single fuzzy simplicial set which captures the relevant topological and
underlying metric structure of the manifold *M*."

Lemma 1 only defines distances between the chosen point **x_i** and its *k*
nearest neighbors; distances between other points **x_j**, **x_k** where i!=k
and i != j are not well-defined. Therefore, the distances between such points
in the space local to **x_i** are defined to be infinite.

The manifold is also assumed to be locally connected.

The above results in the fuzzy topological representation of a dataset:

**Definition** Let *X* = {**x_1**, ..., **x_N**} be a dataset in `R^n`. Let
{(*X*, d_i) | i = 1, ..., N} be a family of extended-pseudo-metric spaces with
common carrier set X such that

d_i(**x_j**, **x_k**) =
- d_*M*(**x_j**, **x_k**) - p if i = j or i = k,
- infinity otherwise

where p is the distance to the nearest neighbor of **x_i** and d_*M* is the
geodesic distance on the manifold *M*, either known *a priori*, or as
approximated per Lemma 1.

The **fuzzy topological representation of *X*** is

- UNION i=1:N FinSing((*X*, d_i)).

Dimensionality reduction can be performed by finding low dimensional
representations that closely match the topological structure of the data.

## Optimizing a low dimensional representation
For *Y* = {**y_i**, ..., **y_n**} a subset of `R^d` (`d << n`), a low
dimensional representation of *X*, we know the manifold and manifold metric
*a priori*, and can compute the fuzzy topological representation directly.
We still include incorporate the distance to the nearest neighbor as per the
local connectedness requirement by supplying a parameter that defines the
expected distance between nearest neighbors in the embedded space.
The fuzzy simplicial set representations of *X* and *Y* can be compared by
converting each to a fuzzy set of edges, given by a reference set *A* and a
membership strength function `mu: A -> [0, 1]`. The sheaf representation is
translated into a classical fuzzy set by ... .
Thus the representations of *X* and *Y* are converted into fuzzy sets, and
compared via the fuzzy set cross entropy. This can be optimized with stochastic
gradient descent as long as the singular set functor `FinSing` is
differentiable.
