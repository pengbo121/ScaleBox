# Install supervisor
sudo python3 -m pip install supervisor

REPO_DIR=$(pwd)

# Configure supervisor
cd supervisor
echo_supervisord_conf > supervisord.conf
printf "[inet_http_server]\nport = 127.0.0.1:9001\n[program:main]\n# Set the command to execute in the specified directory\ndirectory=${REPO_DIR}\n# Here is the startup command of the project you want to manage\ncommand=/bin/make run-distributed\n# Which user to run the process as\n# user=root\n# Automatically apply this when supervisor starts\nautostart=true\n# Automatically restart the process after the process exits\nautorestart=true\n# How long does the process continue to run before it is considered successful\nstartsecs=1\n# number of retries\nstartretries=5\n# stderr log output location\nstderr_logfile=${REPO_DIR}/supervisor/logfile/stderr.log\n# stdout log output location\nstdout_logfile=${REPO_DIR}/supervisor/logfile/stdout.log\n" >> supervisord.conf

# Make dir
mkdir -p logfile

# Clear previous log
> logfile/stderr.log
> logfile/stdout.log

# Close previous supervisor
supervisorctl shutdown > /dev/null 2>&1
# Start supervisor
supervisord -c supervisord.conf
# Check the status of the process
for i in $(seq 1 5); do
    supervisorctl status
    sleep 1
done
