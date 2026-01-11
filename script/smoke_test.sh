#!/bin/bash
URL="http://localhost:5000/"
echo " Running Smoke Test against $URL..."

for i in {1..10}; do
  HTTP_CODE=$(curl --write-out %{http_code} --silent --output /dev/null $URL)
  if [ "$HTTP_CODE" -eq 200 ]; then
    echo " Smoke Test Passed: Server responded with HTTP $HTTP_CODE"
    exit 0
  fi
  echo " Waiting for service to start... (Attempt $i/10)"
  sleep 3
done

echo " Smoke Test Failed: Server did not respond with HTTP 200 after 30 seconds."
exit 1
