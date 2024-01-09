# Dockerfile to build a container for CSE303 development
FROM ubuntu
ENV TZ=America/New_York TROLLTOP=/root/src/ska_src/allh.top SUBMAT=/root/src/ska_src/blosum62
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Initialize repos and upgrade the base system
RUN apt-get update -y
RUN apt-get upgrade -y

# # Install additional software needed for development
# RUN apt-get install -y git man curl build-essential screen gdb libssl-dev psmisc python3


# Change the working directory:
WORKDIR "/root"

COPY ./requirements.txt .

RUN apt-get install -y git man curl build-essential screen gdb libssl-dev psmisc python3 python3-pip 

RUN pip3 install biopython matplotlib networkx grakel


# To make a container from this Dockerfile
# - Pick a name for the image.  We'll go with "cse303"
# - Go to the folder where this Dockerfile exists
# - Type the following: docker build -t cse303 .
#   (note that the last period is part of the command)
#   (note that on Mac and Linux, you might need to prefix the command with 'sudo')

# To launch an instance of the cse303 container
# - You will need to be in PowerShell on Windows, or a terminal on MacOS/Linux
# - Determine the full path to the place on your hard disk where you will save your work
#   (for example /c/Users/yourname/Desktop/cse303 or /home/yourname/cse303)
#   (we will use /home/jones/cse303 for our example)
# - Type the following: docker run --privileged -v /home/jones/cse303:/root --name cse303dev -it cse303

# When your terminal starts up:
# - You will be logged in as root
# - You will be in the root user's home folder (/root)
# - You should see your cse303 folder's contents in there

# When you are done
# - Be sure to type docker rm cse303dev to clean up from your container, so you can start a new one
#   with the same name.