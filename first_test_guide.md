# Guide to first test between SU and RH

The simple example desribed in the README will be used as a first test. To make sure we run the clients in similar environments, Docker will be used.

1. Download the repository to the server.

2. Build the Docker image

   ```cd deceaihealth```
   
   ```docker build -t decaihealth --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile .```

3. Start the Docker container

   ```docker run -d --rm -it --volume $(pwd):/workspace --name decaihealth decaihealth```

4. Execute the container

   ```docker exec -it decaihealth bash```

5. Make sure the server is up and running (someone from SU has to do this). Start a client

   ```python3 client.py --host "hostname" --load_mnist "even"```
   
   where "hostname" is the DNS sent out to the group.

6. The server is configured so that two clients have to be connected beore the training starts. The second client can be started from either institution. To do this from the same computer that started the first client, open a new terminal and do step 4-5 again, with --load_mnist "odd". Now the training should start and the progress should be printed in both terminals.
