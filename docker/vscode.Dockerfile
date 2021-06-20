FROM ayberkydn/deep-learning

# install language related things
RUN pip install black

# add user
ARG USERNAME=user
RUN useradd -ms /bin/bash  $USERNAME
USER $USERNAME

