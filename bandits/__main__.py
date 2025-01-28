import fire
from scripts import plot_results, run_experiments, tabular_test
import jax

jax.config.update("jax_enable_x64", True)

class Experiments:
    def test(self):
        tabular_test.main()

    def plot_experiments(self):
        plot_results.main()

    def run_experiments(self):
        run_experiments.main()

    def run_and_plot(self):
        self.run_experiments()
        self.plot_experiments()


if __name__ == "__main__":
    fire.Fire(Experiments)
