# ChainMaker proof 验证流程

## 1. 生成本地 Hybrid proof 文件

```bash
python3 decentralized_truth_discovery/examples/five_methods_comparison.py
```

默认会输出：

- `EnhancedTruthFinder-Hybrid.proof_bundle.json`
- `hybrid_proof_manifest.json`
- `EnhancedTruthFinder-Hybrid.chainmaker_manifest.json`
- 每个 proof group 对应的：
  - `*.verifySelectedProofFlat.params.json`
  - `*.submitSelectedProofFlat.params.json`

## 2. 编译合约

```bash
scripts/chainmaker/compile_truth_single_proof_registry_n4.sh
```

产物默认在：

- `decentralized_truth_discovery/core/circuits/build/`

## 3. 使用 cmc 部署 EVM 合约

```bash
SDK_CONF_PATH=/path/to/sdk_config.yml \
ABI_PATH=/path/to/TruthSingleProofRegistryN4.abi \
BIN_PATH=/path/to/TruthSingleProofRegistryN4.bin \
ADMIN_KEY_FILE_PATHS=/path/admin1.key,/path/admin2.key,/path/admin3.key \
ADMIN_CRT_FILE_PATHS=/path/admin1.crt,/path/admin2.crt,/path/admin3.crt \
CONTRACT_NAME=truth_single_proof_registry_n4 \
scripts/chainmaker/deploy_truth_single_proof_registry_n4.sh
```

## 4. 只做链上验证

```bash
SDK_CONF_PATH=/path/to/sdk_config.yml \
ABI_PATH=/path/to/TruthSingleProofRegistryN4.abi \
CONTRACT_NAME=truth_single_proof_registry_n4 \
PARAMS_FILE=/path/to/EnhancedTruthFinder-Hybrid.group_0.verifySelectedProofFlat.params.json \
METHOD=verifySelectedProofFlat \
MODE=get \
scripts/chainmaker/verify_hybrid_proof_with_cmc.sh
```

## 5. 验证并把该 proof 提交到链上保存

```bash
SDK_CONF_PATH=/path/to/sdk_config.yml \
ABI_PATH=/path/to/TruthSingleProofRegistryN4.abi \
CONTRACT_NAME=truth_single_proof_registry_n4 \
PARAMS_FILE=/path/to/EnhancedTruthFinder-Hybrid.group_0.submitSelectedProofFlat.params.json \
METHOD=submitSelectedProofFlat \
MODE=invoke \
scripts/chainmaker/verify_hybrid_proof_with_cmc.sh
```

## 6. 在 10.3.242.100 远端执行

如果 `cmc` 和 `sdk_config.yml` 都在远端服务器，可以直接把本地 ABI/bin/params 复制到远端再执行：

### 远端部署

```bash
REMOTE_HOST=gao@10.3.242.100 \
REMOTE_SDK_CONF_PATH=/path/on/server/sdk_config.yml \
REMOTE_ADMIN_KEY_FILE_PATHS=/path/on/server/admin1.key,/path/on/server/admin2.key,/path/on/server/admin3.key \
REMOTE_ADMIN_CRT_FILE_PATHS=/path/on/server/admin1.crt,/path/on/server/admin2.crt,/path/on/server/admin3.crt \
LOCAL_BIN_PATH=/local/path/TruthSingleProofRegistryN4.bin \
LOCAL_ABI_PATH=/local/path/TruthSingleProofRegistryN4.abi \
CONTRACT_NAME=truth_single_proof_registry_n4 \
scripts/chainmaker/remote_deploy_truth_single_proof_registry_n4.sh
```

### 远端验证或提交 proof

```bash
REMOTE_HOST=gao@10.3.242.100 \
REMOTE_SDK_CONF_PATH=/path/on/server/sdk_config.yml \
REMOTE_CONTRACT_NAME=truth_single_proof_registry_n4 \
LOCAL_ABI_PATH=/local/path/TruthSingleProofRegistryN4.abi \
LOCAL_PARAMS_FILE=/local/path/EnhancedTruthFinder-Hybrid.group_0.verifySelectedProofFlat.params.json \
METHOD=verifySelectedProofFlat \
MODE=get \
scripts/chainmaker/remote_verify_hybrid_proof.sh
```

> 注意：远端脚本默认假设服务器上已经安装并可直接执行 `cmc`。若没有，请按 ChainMaker 官方文档先编译/安装 `cmc`。
