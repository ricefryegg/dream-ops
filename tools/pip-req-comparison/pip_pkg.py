# compare 2 pip requirements files, use the first one as bas base, 
# overwrite the second one, and remove items that are not in the first one
def overwrite_requirements(base, target):
    base_reqs_name_ver = {}

    with open(base, 'r') as f:
        base_reqs = f.read().splitlines()
        
        for req in base_reqs:
            if "==" in req:
                req_name, req_ver = req.split("==")
            elif "@" in req:
                req_name, req_ver = req.split(" @ ")

            base_reqs_name_ver[req_name] = req_ver

    
    with open(target, 'r') as f:
        target_reqs = f.read().splitlines()

        for i, req in enumerate(target_reqs):
            if "==" in req:
                req_name, req_ver = req.split("==")
            elif "@" in req:
                req_name, req_ver = req.split(" @ ")

            if base_reqs_name_ver.get(req_name):
                if base_reqs_name_ver.get(req_name) != req_ver:
                    print(f"[different]: {req_name}, {base_reqs_name_ver[req_name]} -> {req_ver}")

                target_reqs[i] = [req_name, base_reqs_name_ver.get(req_name)]
            else:
                target_reqs[i] = [req_name, "none"]
                print(f"[not found]: {req_name}, {req_ver}")
    
    with open(target, 'w') as f:
        for name, ver in target_reqs:
            f.write(f"{name}=={ver}\n")

if __name__ == '__main__':
    overwrite_requirements('base.txt', 'requirements.txt')