from iqcc_cloud_client import IQCC_Cloud

qc = IQCC_Cloud("arbel")

qc.run() # show all ondemand options available

qc.run("openqasm2qua") # show payload for particular ondemand option

r = qc.run("openqasm2qua",
{
"openqasm3": """
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
bit x;
sx $0;
x $0;
rz(pi/2) $1;
barrier $0, $1;
c[0] = measure $0;
c[1] = measure $1;
x = measure $1;

bit[1] c2;
c[0] = measure $1;

if (c[0] == 0) {
sx $1;
} else {
x $1;
}
cz $1, $0;
x $0;
""",
"num_shots": 11,
})

print(r["result"].keys())
print(r["result"]["qua"])


qc.execute(r["result"]["qua"],r["result"]["qua_config"], terminal_output=True)