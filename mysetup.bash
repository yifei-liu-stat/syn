# Setup environment variables for the project
export REPO_DIR=/home/liu00980/Documents/multimodal/tabular/tab-ddpm
cd $REPO_DIR

conda activate tddpm

conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}

conda deactivate
conda activate tddpm



# # Get data
# cd $PROJECT_DIR
# wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
# tar -xvf data.tar