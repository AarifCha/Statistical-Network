<div align="center">

# Monte Carlo using ML

### Aarif C.

</div>

# Inverse Transform Sampling with ML

Monte Carlo relies heavily on inverting a distribution (for example
using the metropolis algorithm). Here we provide an alternative method
to sampling random numbers that aims to reduce the variance of the
integral to produce better results. Lets first state what Monte Carlo
method is. The method approximates the integral of a function by
averaging over the function value at different points to get an
\"average height\".

$$\int_a^b dx f(x) \approx \frac{(b-a)}{N}\sum_{i=1}^N f(x_i)$$

where $x_i$ are random numbers generated according to a uniform
distribution. Rather than using a uniform distribution, we can also
sample for some PDF (probability density function) and divided the
function by the PDF to not bias the integrand by sampling more from
certain areas.

$$\int_a^b dx f(x) \approx \frac{(b-a)}{N}\sum_{i=1, x_i\sim PDF}^N \frac{f(x_i)}{PDF(x_i)}$$

Choosing a PDF that decreases the variance of the integral will produce
better results since the error of the estimation is proportional to the
square root of the variance. This gives us a loss function for a MLP to
be trained on. But before that, we also review a little bit of inverse
transform sampling. Given a CDF (cumulative density function), the
inverse of the function can be used to generate random numbers according
to the corresponding PDF. That is

$$CDF^{-1}(u) = x$$

Where $u$ is a uniform random number and $x$ is a random number
generated from the CDF. We further have that the derivative of the CDF
is the PDF which is required for doing Monte Carlo integration. Now, we
define what our MLP learns. It learns some $CDF^{-1}$ that minimizes the
variance of the integral. So we have

$$NN(u) = CDF^{-1}(u) = x$$

So, it learns to generate random numbers that will be used in Monte
Carlo integration. But we still require the PDF evaluated at the random
number x, $PDF(x)$. To do so, we compute the gradient of the NN to get
the derivative of the $CDF^{-1}(u)$ by running \"in-graph\" gradient
descent (meaning we don't change any of the model parameters). The
derivative of the inverse of a function is give as

$$(f^{-1})'(u) = \frac{1}{f'(f^{-1}(u))}$$

Replacing $f(u)$ with $CDF(U)$ we have

$$(CDF^{-1})'(u) = \frac{1}{CDF'(CDF^{-1}(u))}$$

plugging in $CDF' = PDF$ and $CDF^{-1}(u) = x$, we get our desired
result

$$(CDF^{-1})'(u) = \frac{1}{PDF(x)}$$

The Monte Carlo integral would then look like

$$\int_a^b dx f(x) \approx \frac{(b-a)}{N}\sum_{i=1, x_i = NN(u_i)}^N f(x_i) NN'(u_i)$$

# Joint PDFs

Now we would like similarly results but for multivariate statistics. To
do so lets first review a few concepts of bivariate statistics. Let $X$
and $Y$ be random variables, then we have that the joint $PDF_{XY}(x,y)$
and corresponding $CDF$

$$CDF_{XY}(x,y) = \int_{-\infty}^x \int_{-\infty}^y PDF_{XY}(x',y') dx'dy'$$

Further more, we have marginal and conditional $PDF$s defined as follows

$$\begin{split}
    PDF_{X}(x) &= \int_{-\infty}^{\infty} PDF_{XY}(x,y)dy \\
    PDF_{Y|X}(y|X=x) &= \frac{PDF_{XY}(x,y)}{PDF_X(x)}
\end{split}$$

The conditional $PDF$ is simply the $PDF$ in $Y$ given that the random
variable $X=x$. Similarly, we can also define corresponding $CDF$s as

$$\begin{split}
    CDF_X(x) &= \int_{-\infty}^x PDF_{X}(x')dx' \\
    CDF_{Y|X}(y|X=x) &= \int_{-\infty}^y PDF_{Y|X}(y')dy'
\end{split}$$

Now we can use these functions and repeat Inverse Transform Sampling
method. We first generate a random number according to marginal $CDF$,
$CDF_X(x)$, using a uniform random number $u_1$ by inverting the $CDF$.

$$x = CDF^{-1}_X(u_1)$$

After generating $x$, we can use the conditional $CDF$, $CDF_{Y|X}(y)$,
using another uniform random number $u_2$ and inverting the $CDF$.

$$y = CDF^{-1}_{Y|X}(u_2)$$

The resulting pairs of numbers $(x,y)$ would then be generated according
to $PDF_X(x)PDF_{Y|X}(y|X=x) = PDF_{XY}(x,y)$, which is the desired
output.

The most basic $NN$ implementation would take in $u_1$ and $u_2$, and
output $x$ and $y$. Then by taking gradient's of the functions, we get

$$\begin{split}
    \frac{\partial CDF^{-1}_X(u_1)}{\partial u_1} &= \frac{1}{PDF_X(x)} \\
    \frac{\partial CDF^{-1} _{Y|X}(u_2)}{\partial u_2} &= \frac{1}{PDF _{Y|X}(y)}
\end{split}$$

Which multiplying the two gives us exactly the joint $PDF$.

$$\frac{\partial \, CDF^{-1}_X(u_1)}{\partial u_1} \times \frac{\partial \, CDF^{-1} _{Y|X}(u_2)}{\partial u_2} = \frac{1}{PDF_X(x)PDF _{Y|X}(y)} = \frac{1}{PDF _{XY}(x,y)}$$

Now to ensure the functions that are learned are exactly the inverse of
the marginal and conditional CDFs, we can modify the ML architecture in
many ways. One example would be to have many separate MLPs that are
sequentially tagged together. That is, we would have the first MLP
accept $u_1$ as an input, to output $x$ and then have another MLP that
accepts both $u_2$ and the outputted $x$ to generate $y$. This way we
will mimic the Inverse Transform Sampling much more closely.
