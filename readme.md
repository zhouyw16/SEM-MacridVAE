# Readme


This repo contains Python codes for [Disentangled Representation Learning for Recommendation. IEEE Transactions on Pattern Analysis and Machine Intelligence 2022](https://ieeexplore.ieee.org/abstract/document/9720218/).


## Files

* **main.py**:    main code to train, test and visualize 
* **model.py**:    MultiDAE, MultiVAE, **DisenVAE**(MacridVAE), **DisenEVAE**(SEM-MacridVAE)
* **data.py**:    load dataset and split into train, val_tr, val_te, test_tr, test_te
* **RecomData** 
  * **ml-latest-small**:    sample dataset
    * **prep.py**:    pre-process data
    * **ratings.txt**:    rating actions file
    * **embed.npy**:    image features file
  * **prep.py**:    further process data
* **run**:    save log and model



## Run

```bash
### part 1
./DecomData/<dataset>

### part 2
./DecomData
python prep.py <dataset>

### part 3
./
python main.py --data <dataset> --model <model> --device <cuda>


### Take ml-latest-small dataset as an example: 
./DecomData/ml-latest-small:  python prep.py
./DecomData:                  python prep.py ml-latest-small
# In this repository, the above two steps have been completed, and the results have been saved in the prep directory.
# It is worth noting that we keep users who have at least five rating actions, instead of fifteen in our paper.
# If you would like to keep users who have at least fifteen rating actions, modify the code in line 52 in ./RecomData/prep.py.
./:                           python main.py --data ml-latest-small --model DisenEVAE --device cuda:0
```



