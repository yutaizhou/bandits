import jax
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

a = jnp.ones(7) * 6
b = jnp.ones(7) * 6
b = b.at[0].set(6.8778577)
b = b.at[3].set(6.711237)


@jax.jit
def f(a, b, key):
    x = tfd.InverseGamma(a, b).sample(seed=key)
    return x


key = jr.key(0)
for i in range(10):
    key, subkey = jr.split(key)
    x = f(a, b, subkey)
    print(x.shape)
