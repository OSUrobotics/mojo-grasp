from mojograsp.simcore.path_comparison import Evaluator 
import pathlib

if __name__ == "__main__":
    current_path = pathlib.Path().resolve()
    data_path = current_path.parent
    data_path = data_path.joinpath('demos/rl_demo/data')
    test = Evaluator(str(data_path))
    test.evaluate_single(119)