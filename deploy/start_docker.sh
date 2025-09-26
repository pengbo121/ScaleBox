docker run \
    --privileged \
    -p 8080:8080 \
    -p 8081:8081 \
    --volume ~/icip-sandbox:/icip-sandbox \
    -w /icip-sandbox \
    --health-cmd='python /icip-sandbox/deploy/a_plus_b.py || exit 1' \
    --health-interval=2s \
    -itd \
    --restart unless-stopped \
    zhengxin1999/icip-sandbox:v1 \
    make run-online

