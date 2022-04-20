import warnings
from argparse import ArgumentParser
import flwr as fl

# -----------------------------------------------------------------------------
# "Main" function.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--rounds", type=int, default=5)  
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=str, default='8080')
    args = parser.parse_args()

    # Start federated server
    fl.server.start_server(args.host + ':' + args.port, config={"num_rounds": args.rounds})
