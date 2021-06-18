FROM ayberkydn/deep-learning

# add user
ARG USERNAME=user
RUN useradd -ms /bin/bash  $USERNAME
USER $USERNAME

