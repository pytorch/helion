# Fixed-configuration diagnostics

These experiments are not cold-autotune goal results. All candidates below are rejected; identity runs validate the harness. Sources and before/after FP64, 30-repeat, bitwise and immutability audits were checked from each passing raw result.

| Experiment | Shape | Original us | Candidate us | CAKE us | Candidate/original |
|---|---:|---:|---:|---:|---:|
|[identity18-timing](probes/identity18-timing/result.json)|18|438.784|438.784|424.448|1.0000|
|[outputdrain18-timing](probes/outputdrain18-timing/result.json)|18|439.040|668.416|424.704|1.5224|
|[outputdrain18-tma7-timing](probes/outputdrain18-tma7-timing/result.json)|18|439.040|529.120|424.672|1.2052|
|[projection1-v2-timing](probes/projection1-v2-timing/result.json)|1|54.016|56.064|56.064|1.0379|
|[projection18-v2-timing](probes/projection18-v2-timing/result.json)|18|439.040|459.520|424.672|1.0466|
|[projection18-v3-timing](probes/projection18-v3-timing/result.json)|18|438.784|457.216|424.448|1.0420|
|[unroll18-timing](probes/unroll18-timing/result.json)|18|438.352|467.712|423.680|1.0670|
|[unroll18-v2-timing](probes/unroll18-v2-timing/result.json)|18|439.040|473.856|424.704|1.0793|
