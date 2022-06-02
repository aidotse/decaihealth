# Guide to ISIC test between SU and RH

This describes step by step how to perform the federated ISIC test between SU and RH. The model trained will be a classifier, able to separate images of moles from malignant and benign ones. The experiment was developed by Sandra Carrasco Limeros and Sylwia Majchrowska.

The first steps runs the experiment locally, and can be done separately by both organisation to check that everything is up and running. In the last steps, both organisations are needed. 

## Preparations

1. Download the repository here https://github.com/aidotse/decentralizedAI_dermatology and rename it decentralized_ai_dermatology.

2. Download the dataset here https://www.kaggle.com/datasets/nroman/melanoma-external-malignant-256 and place it in /decentralized_ai_dermatology/data and unzip it.

3. Use the provided Dockerfile to build an image with the required library dependencies, listed in requirements.txt.

    ```cd decentralized_ai_dermatology```

    ```docker build -t decentralized_ai_dermatology --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile .```

4. Start the Docker container

    ```docker run -d --rm -it --volume $(pwd):/workspace --shm-size 16G --gpus all --name decentralized_ai_dermatology decentralized_ai_dermatology```

5. Execute the container

    ```docker exec -it decentralized_ai_dermatology bash```

## Individual tests

6. To verify that everything is working, train the model locally, without any federated learning. This will train the classifier on 5% of the data. 

    ```python train_local.py --path_data "/workspace/data" --nowandb --num_partitions 20 --partition 0```

7. Next, try the experiment with a local server. Start a local server

    ```python server.py --nowandb```

8. Start two clients that connects with the local server. Do this in the same container as the local server but in separate terminals (change to --partition 1 for the second client).

    ```python client_isic.py --path "/workspace/data" --num_partitions 2 --partition 0 --nowandb --batch_train 16```

    The server is configured so that two clients have to be connected beore the training starts. When both clients are started, the training should start and the progress should be printed in both terminals.

9. Now that you know everything works with a local server, try running two clients that connects with the common server. Make sure the server is up and running (someone from SU has to do this). Try the connection to the server by 

    ```curl "hostname":8080```

    where "hostname" is the DNS sent out to the group. This should return something like "curl: (1) Received HTTP/0.9 when not allowed". Next, start a client by running

    ```python client_isic.py --path "/workspace/data" --num_partitions 2 --partition 0 --nowandb --host "hostname" --batch_train 16```

    Start a second client by running the above again, in the same container but in a different terminal, and change the partition.

## Joint test

10. The final step is to start one client from each organization. This is done exactly as in step 9, with some coordination. 