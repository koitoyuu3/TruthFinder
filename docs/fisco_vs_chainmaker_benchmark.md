# FISCO BCOS 与 ChainMaker 全流程执行时间对比实验文档

## 1. 实验做了什么

本实验面向一个“链上请求 + 链下 AI 执行 + 结果回调 + 链上 proof 验证”的完整闭环系统，比较同一业务流程在 FISCO BCOS 和 ChainMaker 两条区块链上的端到端执行时间。实验关注的不是大模型推理速度，而是区块链部分在完整业务闭环中的执行开销，因此采用固定 AI 输出文件替代实时推理，保证两条链输入一致、回调内容一致、proof 内容一致，从而把变量集中在区块链请求、回调写入、结果查询和 proof 链上证明阶段。

完整流程包括：用户发起请求并写入 OracleAggregator 合约；系统读取链上请求信息；将固定 AI 结果输出并依次回调写入链上；查询链上聚合结果与完成状态；读取 SemanticTruth 生成的 Schnorr NIZK proof bundle；调用 TruthSchnorrProofRegistry 合约在链上验证 proof；将 proof 相关摘要登记到链上；最后查询 proof submission 是否存在。只有当 proof verifier 在链上返回成功，并且提交后的 submission 能够在链上查到时，本次实验才计为 proof 成功。

实验共进行了 10 轮重复测试。每轮分别在 ChainMaker 和 FISCO BCOS 上执行一次相同流程，共得到 20 份原始结果 JSON。20 次结果均满足 `completed=true`、`proof.status=proved-on-chain`、`proof.onChainProved=true`。

## 2. 数据来源与数据规模

该结果文件包含 50 个 agent 对 300 条新闻的输出。本次链上回调阶段为了贴近合约的 4 节点 oracle 设计，选取 `agent-1` 至 `agent-4` 的输出作为 4 个 oracle 节点回调内容。每个 agent 对应 300 条新闻摘要，按第 1 条到第 300 条新闻的原始顺序排列。为了避免把 4.8 MB 完整结果直接写入链上，实验服务只提取每个 agent 的 `response` 字段，形成 `responses only, original news order` 的 JSON payload，再使用 gzip + base64 编码写链。单个 agent 的原始 payload 约 55,111 bytes，压缩编码后约 31,492 到 31,496 bytes。

proof 数据文件为：

```text
/home/gao/InsdustrialInternet/SemanticTruth.text.full_proof_payload_bundle.json
```

该文件大小为 1,300,834 bytes，SHA-256 为：

```text
53a2570fb108fd524836412792e88d1925bcd276f6c3f5f22f468abdcca41af4
```

proof bundle 的关键信息如下：

```text
method: SemanticTruth
rowCount: 1500
proofGroupCount: 1
proofScheme: schnorr_nizk_v1
proofPayloadPresent: true
proofId: 0x597f55efbc028e5593d578b268c23a2f0f0a497d4bca4822d41c2053815a79e9
statementDigestSha256: 0xe52bb4ec2a05942a3c5a9875f8205d339d4f412f0daecc3af724058322e13bd8
resultDigest: 0x0cc0e9149331a98788ac6e1270c35d5a25c9003c07051cd21cbdf9f44641501a
```

这里的 `rowCount=1500` 对应 50 个 agent 与 300 条新闻的组合规模，即 `50 x 300 = 1500` 条 truth discovery 输入记录。

## 3. 实验设计与实验流程

实验目标是比较两条链在相同业务负载下完成全流程的耗时。为了控制变量，本实验固定以下输入：

- 相同 prompt：`benchmark fixed AI result request`
- 相同固定 AI 结果文件：`result_request15_llama31_8b.json`
- 相同 proof bundle：`SemanticTruth.text.full_proof_payload_bundle.json`
- 相同 oracle 节点数量：4 个节点，对应 `agent-1` 到 `agent-4`
- 相同 payload 编码方式：`agent-responses-json-gzip-base64`
- 相同 proof scheme：`schnorr_nizk_v1`
- 相同重复次数：每条链 10 次

实验后端暴露的 benchmark 接口为：

```text
POST /api/benchmark/ollama-fixed-result
```

调用脚本为：

```text
/home/gao/InsdustrialInternet/scripts/run_fixed_result_benchmark.sh
```

脚本核心参数如下：

```bash
BACKEND_URL=http://127.0.0.1:8080
RESULT_PATH=/home/gao/InsdustrialInternet/result_request15_llama31_8b.json
PROOF_BUNDLE_PATH=/home/gao/InsdustrialInternet/SemanticTruth.text.full_proof_payload_bundle.json
CHAINS="chainmaker fisco"
OUTPUT_DIR=/home/gao/InsdustrialInternet/benchmark-results
```

每轮实验对每条链执行以下步骤：

1. 请求上链：调用 OracleAggregator 合约发起请求，写入 prompt，并获得链上 `requestId`。
2. 请求查询：根据 `requestId` 查询链上请求元信息，确认请求已记录。
3. 固定结果加载：读取固定 AI 输出文件，从 50 个 agent 输出中提取 `agent-1` 至 `agent-4`。
4. payload 生成：每个 agent 取 300 条新闻对应的 `response` 字段，形成只包含摘要结果的 JSON，再 gzip + base64 编码。
5. 回调写链：模拟 4 个 oracle 节点依次调用 `submitOllamaResult`，把 4 份 agent payload 写回链上。
6. 结果查询：查询链上 4 个节点的回调结果，计算 `chainResultDigest`，并查询最终完成状态。
7. proof 读取：读取 full proof bundle，提取 Schnorr proof 的 `public_key`、`commitment`、`challenge`、`response` 以及 statement/result/proof digest。
8. proof 链上验证：调用 `TruthSchnorrProofRegistry.verifyProof`，验证 Schnorr NIZK 证明。
9. proof 链上提交：调用 `TruthSchnorrProofRegistry.submitProof`，提交 proof 摘要和验证输入。该函数内部会再次调用 `verifyProof`，验证通过后登记 `submissionKey`。
10. proof 提交查询：调用 `hasSubmission(submissionKey)`，确认 proof 已经在链上登记。
11. 结果保存：每次链上运行输出一份 JSON，10 轮结束后生成 summary JSON。

本实验中 proof 合约不是简单保存 proof 文件，而是在链上执行 Schnorr 验证逻辑。验证逻辑包括：

```text
expectedChallenge = sha256(challengeMaterial)
检查 challenge == expectedChallenge
计算 g^response mod p
计算 commitment * publicKey^challenge mod p
比较两边是否相等
```

其中 `p` 为 2048-bit MODP subgroup 参数，`public_key`、`commitment`、`response` 均按 256 bytes 大整数输入。`submitProof` 只有在 `verifyProof` 返回 true 后才会写入 `submittedAt[submissionKey]` 并触发 `SchnorrProofSubmitted` 事件。

## 4. 实验结果

10 轮结果全部成功：

| 链 | 运行次数 | 成功次数 | proof 状态 |
|---|---:|---:|---|
| ChainMaker | 10 | 10 | `proved-on-chain` |
| FISCO BCOS | 10 | 10 | `proved-on-chain` |

平均耗时如下，单位均为毫秒：

| 指标 | ChainMaker 平均值 | FISCO BCOS 平均值 |
|---|---:|---:|
| 总耗时 `totalNoAiMs` | 4786.7 | 2627.4 |
| 请求上链 `requestWriteMs` | 520.5 | 25.8 |
| prompt 查询 `promptQueryMs` | 9.3 | 2.5 |
| 回调写链 `callbackWriteMs` | 2107.1 | 2058.3 |
| 结果查询 `resultQueryMs` | 63.2 | 14.9 |
| final meta 查询 `finalMetaQueryMs` | 8.9 | 2.3 |
| proof 验证 `proofVerifyMs` | 523.7 | 39.3 |
| proof submissionKey 计算 `proofSubmissionKeyMs` | 8.8 | 2.5 |
| proof 提交 `proofSubmitMs` | 1536.5 | 147.4 |
| proof 提交查询 `proofHasSubmissionMs` | 7.8 | 2.2 |
| proof 总耗时 `proofTotalMs` | 2076.6 | 191.5 |

从端到端结果看，ChainMaker 平均总耗时为 4786.7 ms，FISCO BCOS 平均总耗时为 2627.4 ms。在本实验环境和当前实现下，FISCO BCOS 的全流程耗时约为 ChainMaker 的 54.9%，或者说 ChainMaker 平均约为 FISCO BCOS 的 1.82 倍。

从分阶段结果看，两条链在回调写链阶段非常接近：ChainMaker 平均 2107.1 ms，FISCO BCOS 平均 2058.3 ms，差距约 48.8 ms。真正拉开差距的是请求上链和 proof 阶段：ChainMaker 请求上链平均 520.5 ms，而 FISCO BCOS 为 25.8 ms；ChainMaker proof 总耗时平均 2076.6 ms，而 FISCO BCOS 为 191.5 ms。

## 5. 指标含义

`totalNoAiMs` 表示不包含实时 AI 推理的完整链路总耗时。它从请求上链前开始计时，到 proof 验证、提交和提交查询完成后结束。虽然名称中有 `NoAi`，但它包含链上请求、链上回调、链上查询和链上 proof 操作，只是不包含 Ollama / LLM 实时生成答案的耗时。

`requestWriteMs` 表示发起请求写链耗时。该阶段调用 OracleAggregator 的请求函数，将 prompt 写入链上并返回 requestId。它反映一次普通业务请求交易从 SDK 发起到链端返回的耗时。

`promptQueryMs` 表示请求元信息查询耗时。该阶段根据 requestId 查询链上保存的 prompt、响应数量和完成状态等信息。它一般是只读调用，因此通常远低于写交易。

`callbackWriteMs` 表示 4 个 oracle 节点回调写链的总耗时。本实验依次写入 `agent-1` 到 `agent-4` 的 payload，每个 payload 是 300 条新闻摘要 response 的 gzip+base64 字符串，大小约 31 KB。该指标包含 4 次写链操作的累计时间。

`callbackNodeMs` 表示每个 oracle 节点单独回调写链耗时。它记录每个 nodeIndex 的 `elapsedMs`、payload 大小和 payload SHA-256，可用于判断是否某个节点回调异常变慢。

`resultQueryMs` 表示查询链上完整回调结果的耗时。该阶段读取 4 个节点的结果和 completed 状态，并用于计算 `chainResultDigest`。

`finalMetaQueryMs` 表示最终元信息查询耗时。它在所有回调完成后再次查询 respondedCount 和 completed，用来确认链上状态已经进入完成状态。

`respondedCount` 表示链上已提交结果的 oracle 节点数量。本实验期望为 4。

`completed` 表示链上 OracleAggregator 是否认为请求已完成。本实验只有 4 个节点结果都写入后才应为 true。

`chainResultDigest` 表示链上结果的摘要。本实验用它检查两条链读取到的回调结果是否一致，避免只比较耗时而忽略结果内容。

`proofVerifyMs` 表示单独调用 `TruthSchnorrProofRegistry.verifyProof` 的耗时。该函数执行 challenge 校验和 Schnorr 等式验证，是纯验证阶段。

`proofSubmissionKeyMs` 表示计算或查询 submissionKey 的耗时。submissionKey 由 `runIdHash`、`experimentId`、`groupId` 和 `proofId` 组合计算，用于唯一标识一次 proof 提交。

`proofSubmitMs` 表示调用 `submitProof` 的耗时。该阶段不仅写入 proof 摘要，还会在合约内部再次调用 `verifyProof`。因此 proof 总成本中实际包含两次链上证明验证：一次显式 `verifyProof`，一次 `submitProof` 内部验证。

`proofHasSubmissionMs` 表示调用 `hasSubmission(submissionKey)` 查询 proof 是否已登记的耗时。只有该值返回 true，才说明 proof 提交在链上可查询。

`proofTotalMs` 表示 proof 阶段的总耗时，包含 `verifyProof`、`submissionKey`、`submitProof` 和 `hasSubmission`。它是衡量链上 proof 验证与登记成本的核心指标。

`proof.status=proved-on-chain` 表示链上 proof 验证、提交和提交查询均成功。

`proof.onChainProved=true` 表示本次结果不是本地验证或模拟结果，而是已经经过链上 verifier 并完成链上登记。

## 6. 实验结果的理论分析

本实验中 FISCO BCOS 的总耗时显著低于 ChainMaker，但不能简单解释为“FISCO BCOS 在任何场景下都一定更快”。更准确的解释是：在当前实验部署、SDK 调用路径、EVM 合约执行环境和 proof 合约实现下，FISCO BCOS 在普通请求写入和大整数 proof 验证两个关键阶段表现出更低开销。

第一，回调写链不是主要差距来源。两条链的 `callbackWriteMs` 分别为 2107.1 ms 和 2058.3 ms，差距很小。这说明当 payload 较大且需要连续写入 4 个节点结果时，两条链在大 payload 写入阶段的表现接近。该阶段主要受 payload 体积、交易编码、合约存储写入和顺序提交 4 次交易影响。由于每次写入的数据约 31 KB，4 次累计超过 120 KB，数据传输和链上存储成本占比较大，因此两条链都需要约 2 秒左右。

第二，请求上链阶段差距明显。ChainMaker 的 `requestWriteMs` 平均为 520.5 ms，而 FISCO BCOS 仅为 25.8 ms。请求上链写入的数据很小，主要是 prompt 和请求元信息，因此该阶段更能反映 SDK 调用链路、交易提交路径、节点连接状态和链端交易确认机制的基础开销。在本实验部署中，FISCO BCOS 后端连接的是本机 `127.0.0.1:20200/20201` RPC 节点，链路较短；ChainMaker 通过 Java SDK 走 gRPC/TLS 链路，历史启动日志中还出现过部分 ChainMaker 节点 TLS 证书校验失败，SDK 最终使用可用节点继续运行。这类连接管理、TLS 握手、节点选择和 gRPC 调用链路会增加请求写入的固定开销。因此在小交易场景下，ChainMaker 的固定开销更容易被放大。

第三，proof 阶段是总耗时差距的核心来源。ChainMaker 的 `proofTotalMs` 平均为 2076.6 ms，FISCO BCOS 为 191.5 ms，相差约 10.84 倍。进一步拆分可见，ChainMaker 的 `proofVerifyMs` 为 523.7 ms，`proofSubmitMs` 为 1536.5 ms；FISCO BCOS 对应为 39.3 ms 和 147.4 ms。proof 合约执行的是 2048-bit 模群上的 Schnorr 验证，包含大整数解析、模幂运算、模乘运算和等式比较。该类计算在 EVM 中不是普通存储写入，而是高计算密度的合约执行任务。不同链的 EVM 实现、预编译支持、虚拟机执行优化、gas 计量策略和交易执行管线都会显著影响该阶段耗时。

第四，`submitProof` 本身比 `verifyProof` 更慢是符合预期的。本实验为了明确区分“链上验证是否成功”和“证明是否登记成功”，先调用一次 `verifyProof`，再调用 `submitProof`。而 `submitProof` 内部为了防止无效 proof 被登记，会再次执行 `verifyProof`，随后计算 submissionKey、检查是否重复提交、写入 `submittedAt` 并触发事件。因此 `proofSubmitMs` 理论上应当大于单独的 `proofVerifyMs`。实验中 ChainMaker 的 `submitProof` 为 1536.5 ms，明显大于 `verifyProof` 的 523.7 ms；FISCO BCOS 的 `submitProof` 为 147.4 ms，也明显大于 `verifyProof` 的 39.3 ms，符合这一设计逻辑。

第五，查询类指标差距较小但方向一致。FISCO BCOS 在 `promptQueryMs`、`resultQueryMs`、`finalMetaQueryMs` 和 `proofHasSubmissionMs` 上均低于 ChainMaker。查询操作通常不产生共识写入，更多体现 SDK 到节点的读路径、合约 ABI 解码和节点本地状态读取效率。在当前部署下，FISCO BCOS 的本机 RPC 路径更短，因此查询耗时也更低。

第六，本实验用固定 AI 输出是必要的控制变量。如果每轮都实时调用 Ollama 或其他大模型，则总耗时会被模型推理波动主导，难以分辨区块链系统自身的性能差异。固定结果测试把 AI 推理从计时中剥离，同时保留真实业务中的回调 payload、链上结果存储和 proof 验证，使实验更适合回答“同一链上闭环流程在 FISCO BCOS 和 ChainMaker 上分别需要多少时间”这个问题。

第七，结果说明当前系统的性能瓶颈主要有两个：大 payload 多节点回调写链，以及 ChainMaker 上的 proof 合约验证。前者在两条链上都接近 2 秒，是系统共有成本；后者在 ChainMaker 上约 2.08 秒，在 FISCO BCOS 上约 0.19 秒，是两链差异的主要来源。如果要优化整体系统，可以考虑三类方向：减少回调上链 payload，只上链摘要而将完整结果放到链下可验证存储；将 proof 验证拆成更适合链上执行的电路或预编译形式；在实验设计上区分“显式 verify + submit 内部 verify”的双验证模式和“只 submit 一次验证”的生产模式。

第八，论文中应谨慎表述实验结论。本实验可以支持的结论是：在本实验环境、当前合约实现和固定 AI 输出负载下，FISCO BCOS 的全流程平均耗时低于 ChainMaker，主要优势体现在请求上链和 Schnorr proof 链上验证/提交阶段；两条链在 4 个 agent 大 payload 回调写链阶段差距较小。该结论不应泛化为所有合约、所有网络拓扑和所有部署条件下的绝对性能排序。

ChainMaker proof registry 合约名为：

```text
TruthSchnorrProofRegistry
```

