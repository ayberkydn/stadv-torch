
FROM ayberkydn/deep-learning

# add user
ARG USERNAME=ayb
RUN useradd -ms /bin/bash  $USERNAME
USER $USERNAME
