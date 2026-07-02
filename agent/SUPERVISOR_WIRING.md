# Wiring the Supervisor Agent (remaining steps)

Everything except the final MCP-connection + Supervisor creation is already
provisioned and verified on workspace `adb-8333330282859393.13`:

| Resource | Value |
|----------|-------|
| Catalog / schema | `supply_chain_stress_test.data` |
| Tables | `nodes` (35), `edges` (44), `bom` (33) — with column comments |
| Genie space | **`01f175ca4bd61e39a5ebe32c4f3dff73`** ("Supply Chain Network") |
| MCP app | **`supply-chain-mcp`** → `https://supply-chain-mcp-8333330282859393.13.azure.databricksapps.com` |
| MCP endpoint | `POST /api/mcp` (JSON-RPC 2.0), tool `run_supply_chain_stress_test` |
| App service principal | `7a85397e-9d84-41e9-a5ec-e01a78c888ee` (id `143020438071390`) |
| Warehouse | `a54c6569f699db53` (attached to app as `sql-warehouse`) |

The app SP already has `USE_CATALOG` / `USE_SCHEMA` / `SELECT` on the catalog, and
the resolver has been verified end-to-end against the live tables (reconstruct →
pyomo TTR/TTS matches the original dataset exactly).

The one remaining gate is a **service-principal OAuth secret** for the UC HTTP
(MCP) connection — a real long-lived credential, so it is left for you to authorize.

---

## Step 1 — Mint an OAuth (M2M) secret for the app service principal

```bash
databricks service-principal-secrets create 143020438071390 --profile DEFAULT --output json
# capture "id" and "secret" from the output
```

The `client_id` for the connection is the app SP's application id:
`7a85397e-9d84-41e9-a5ec-e01a78c888ee`. The `client_secret` is the `secret` value above.

## Step 2 — Grant the SP `CAN USE` on the app

Via the Apps UI (Permissions → add the SP with **Can use**), or the App permissions API.
This lets the SP's OAuth token pass the app's auth proxy.

## Step 3 — Create the UC HTTP MCP connection

```sql
CREATE CONNECTION supply_chain_mcp TYPE HTTP
OPTIONS (
  host 'https://supply-chain-mcp-8333330282859393.13.azure.databricksapps.com',
  port '443',
  base_path '/api/mcp',
  client_id '7a85397e-9d84-41e9-a5ec-e01a78c888ee',
  client_secret '<secret from step 1>',
  oauth_scope 'all-apis',
  token_endpoint 'https://adb-8333330282859393.13.azuredatabricks.net/oidc/v1/token',
  is_mcp_connection 'true'
);
```

## Step 4 — Test the connection before wiring

```sql
SELECT http_request(
  conn => 'supply_chain_mcp',
  method => 'POST',
  path => '',
  json => '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
);
-- expect a result listing run_supply_chain_stress_test

SELECT http_request(
  conn => 'supply_chain_mcp',
  method => 'POST',
  path => '',
  json => '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"run_supply_chain_stress_test","arguments":{"disrupted":["T2_4"],"ttr":6}}}'
);
-- expect content text containing lost_profit=..., tts=..., and both optimized_network_* strings
```

If `tools/list` fails with an auth error, re-check Step 2 (SP needs Can use on the app).

## Step 5 — Create the Supervisor Agent

Via the `manage_mas` MCP tool (or the Agent Bricks UI):

```python
manage_mas(
    action="create_or_update",
    name="Supply Chain Supervisor",
    agents=[
        {
            "name": "genie_data_agent",
            "genie_space_id": "01f175ca4bd61e39a5ebe32c4f3dff73",
            "description": (
                "Answers data-lookup questions about the supply chain network: "
                "node inventory, production capacity, demand, profit margins, "
                "which supplier produces which material, supply links between "
                "tiers (downstream/upstream sites), and bill-of-materials. Use "
                "for any factual question about the network's structure or "
                "current operational parameters."
            ),
        },
        {
            "name": "optimization_agent",
            "connection_name": "supply_chain_mcp",
            "description": (
                "Runs a supply chain stress-test optimization for a disruption "
                "scenario (one or more nodes going down for a given time-to-recover). "
                "Returns profit loss during recovery, time-to-survive, and the "
                "optimized network with vs. without the disruption. Use for ANY "
                "question about what happens if a node fails, how long recovery "
                "takes, resulting losses, or mitigation recommendations."
            ),
        },
    ],
    description="Supply chain assistant: routes data questions to Genie and disruption/stress-test questions to the pyomo optimization MCP server.",
    instructions=(
        "Route data-lookup questions (inventory, capacity, demand, margins, "
        "material types, supply links, bill of materials) to genie_data_agent. "
        "Route disruption / failure / 'goes down' / time-to-recover (TTR) / "
        "recovery / mitigation / 'what should I do' questions to optimization_agent. "
        "Node IDs look like T1_5 (product), T2_4 (direct supplier), T3_10 (sub-supplier)."
    ),
    examples=[
        {"question": "List all downstream production sites for the raw material supplied by T3_10.",
         "guideline": "Route to genie_data_agent."},
        {"question": "What happens if T2_4 goes down and takes 6 weeks to recover? What should I do?",
         "guideline": "Route to optimization_agent."},
        {"question": "Tell me the demand for T1_5 and the inventory of all materials needed to produce it.",
         "guideline": "Route to genie_data_agent."},
        {"question": "There has been an incident at T3_15 and it will go down for 10 time units. How do I mitigate the risk?",
         "guideline": "Route to optimization_agent."},
    ],
)
```

Then grant the supervisor's service principal `USE CONNECTION` on `supply_chain_mcp`
if prompted, and poll `manage_mas(action="get", tile_id=...)` until the endpoint is `ONLINE`.

## Step 6 — Evaluate

Open `agent/02b_evaluate_supervisor.ipynb`, set `MAS_ENDPOINT_NAME` to the
supervisor's serving endpoint, and run it. The routing scorer checks that data
questions hit `genie_data_agent` and disruption questions hit `optimization_agent`.
