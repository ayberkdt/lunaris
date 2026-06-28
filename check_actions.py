import urllib.request, json
req = urllib.request.Request('https://api.github.com/repos/ayberkdt/lunaris/actions/runs/28322818271/jobs')
try:
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode())
    guardrails_job = next(j for j in data['jobs'] if j['name'] == 'guardrails')
    print("Guardrails Job ID:", guardrails_job['id'])
    
    # Try fetching log
    log_req = urllib.request.Request(f"https://api.github.com/repos/ayberkdt/lunaris/actions/jobs/{guardrails_job['id']}/logs")
    log_resp = urllib.request.urlopen(log_req)
    log_text = log_resp.read().decode('utf-8')
    print("--- LOG START ---")
    lines = log_text.splitlines()
    for line in lines[-150:]: # print last 150 lines
        print(line)
    print("--- LOG END ---")
except Exception as e:
    print("Error:", e)
