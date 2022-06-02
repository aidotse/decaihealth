# Guide to first test between SU and RH

The simple example described in the README will be used as a first test. To make sure we run the clients in similar environments, Docker will be used.

The first steps runs the experiment locally, and can be done separately by both organisation to check that everything is up and running. In the last steps, both organisations are needed. 

## Preparations

1. Download the repository to the server.

2. Use the provided Dockerfile to build an image with the required library dependencies, listed in requirements.txt.

   ```cd deceaihealth```
   
   ```docker build -t decaihealth --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile .```

3. Start the Docker container

   ```docker run -d --rm -it --volume $(pwd):/workspace --gpus all --name decaihealth decaihealth```

4. Execute the container

   ```docker exec -it decaihealth bash```

## Individual tests

5. To verify that everything is working, train the model locally, , without any federated learning.

   ```python client.py --locally --load_mnist all --epochs 15```

6. Next, try the experiment with a local server. Start a local server

   ```python server.py --rounds 15 --host <server-dns>```

7. Start two clients that connects with the local server. Do this in the same container as the local server but in separate terminals (change to --load_mnist odd for the second client).

   ```python3 client.py --load_mnist even```

   The server is configured so that two clients have to be connected beore the training starts. When both clients are started, the training should start and the progress should be printed in both terminals.


8. Now that you know everything works with a local server, try running two clients that connects with the common server. Make sure the server is up and running (someone from SU has to do this). Try the connection to the server by 

    ```curl "hostname":8080```

    where "hostname" is the DNS sent out to the group. This should return something like "curl: (1) Received HTTP/0.9 when not allowed". Next, start the clients by running

   ```python3 client.py --load_mnist even --host "hostname"```

    Start a second client by running the above again, in the same container but in a different terminal, and change to --load_mnist odd.

## Joint test

9. The final step is to start one client from each organization. This is done exactly as in step 8, with some coordination. 
