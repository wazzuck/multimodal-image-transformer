echo "--- Installing dependencies ---"
# Activate base env (might be needed depending on conda init)
# source "$HOME/miniconda3/bin/activate" base # Or just rely on conda init

conda install -y -c conda-forge pip

# Install requirements
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing packages from requirements.txt. Exiting."
    exit 1
fi

if [ "$HOSTNAME" != "penguin" ]; then
    echo "Installing tmux via conda..."
    conda install -y -c conda-forge tmux # Add -y for non-interactive install
    if [ $? -ne 0 ]; then
        echo "Error installing tmux via conda. Exiting."
        exit 1
    fi
fi

apt-get update
apt-get install vim
