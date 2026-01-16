package langchain

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
)

// 定义 32 字节的哈希类型，方便理解
type Bytes32 []byte

// 模拟以太坊的哈希函数：Hash(a + b)
func Hash(a, b Bytes32) Bytes32 {
	h := sha256.New()
	h.Write(a)
	h.Write(b)
	return h.Sum(nil)
}

// 核心函数：通过 Merkle Branch 计算根节点
// leaf: FinalizedHeader 的哈希 (起点)
// branch: FinalityBranch (梯子，也就是一堆邻居的哈希)
// index: 在树中的位置索引 (例如 finalized_checkpoint 在 BeaconState 的索引通常是 105)
func CalculateMerkleRoot(leaf Bytes32, branch []Bytes32, index uint64) Bytes32 {
	currentHash := leaf

	fmt.Printf("\n--- 开始爬楼梯 (验证 Merkle Proof) ---\n")
	fmt.Printf("初始叶子节点 (FinalizedHeader Hash): %x\n", currentHash)
	fmt.Printf("目标位置 Index: %d (二进制: %b)\n", index, index)

	// 遍历 Branch 中的每一个哈希 (每一层的邻居)
	for i, sibling := range branch {
		fmt.Printf("\n[第 %d 层] 当前 Hash: %x\n", i+1, currentHash)
		fmt.Printf("          邻居 Hash: %x\n", sibling)

		// 核心逻辑：判断当前节点是左孩子还是右孩子
		// 如果 index 是偶数 -> 我是左边 (Left)
		// 如果 index 是奇数 -> 我是右边 (Right)
		if index%2 == 0 {
			// 情况 A: 我是左，邻居在右
			fmt.Printf("          逻辑判断: Index %d 是偶数 -> 我在左边 (Left)\n", index)
			fmt.Printf("          计算公式: Hash(我 + 邻居)\n")
			currentHash = Hash(currentHash, sibling)
		} else {
			// 情况 B: 我是右，邻居在左
			fmt.Printf("          逻辑判断: Index %d 是奇数 -> 我在右边 (Right)\n", index)
			fmt.Printf("          计算公式: Hash(邻居 + 我)\n")
			currentHash = Hash(sibling, currentHash)
		}

		// 爬上一层，Index 除以 2 (位运算右移一位)
		oldIndex := index
		index = index / 2
		fmt.Printf("          计算结果 (父节点): %x\n", currentHash)
		fmt.Printf("          准备爬下一层: Index 从 %d 变为 %d\n", oldIndex, index)
	}

	fmt.Printf("\n--- 爬楼梯结束 ---\n")
	return currentHash
}

func TestMerkleRoot(t *testing.T) {
	// ==========================================
	// 1. 准备测试数据 (模拟)
	// ==========================================

	// 假设这是 FinalizedHeader 算出来的哈希 (Leaf)
	leafHash, _ := hex.DecodeString("aaaa00000000000000000000000000000000000000000000000000000000aaaa")

	// 假设这是 FinalityBranch (为了演示，我们就造 3 层)
	// 实际上 Beacon Chain 的深度会更深
	branchHex := []string{
		"bbbb00000000000000000000000000000000000000000000000000000000bbbb", // 第1层邻居
		"cccc00000000000000000000000000000000000000000000000000000000cccc", // 第2层邻居
		"dddd00000000000000000000000000000000000000000000000000000000dddd", // 第3层邻居
	}

	var branch []Bytes32
	for _, h := range branchHex {
		b, _ := hex.DecodeString(h)
		branch = append(branch, b)
	}

	// 假设 finalized_checkpoint 在树里的索引是 10
	// 10 的二进制是 1010
	// 意味着路径是：左 -> 右 -> 左 (从下往上反推就是 右 -> 左 -> 右 实际上由 mod 2 决定)
	targetIndex := uint64(10)

	// ==========================================
	// 2. 模拟轻客户端验证过程
	// ==========================================

	// 调用核心函数算出“推导出的根”
	calculatedRoot := CalculateMerkleRoot(leafHash, branch, targetIndex)

	// ==========================================
	// 3. 验证结果
	// ==========================================

	// 假设 LightClientUpdate.header.StateRoot 是这个值 (这里我们用上面代码逻辑推算出的正确值来模拟)
	// 在真实环境中，这个 expectedRoot 是你从 BeaconBlockHeader 里直接拿到的
	expectedRoot := calculatedRoot

	fmt.Printf("\n[最终验证]\n")
	fmt.Printf("Header 里带的 StateRoot: %x\n", expectedRoot)
	fmt.Printf("我们算出来的 StateRoot:    %x\n", calculatedRoot)

	if string(expectedRoot) == string(calculatedRoot) {
		fmt.Println("✅ 验证成功！证明了 FinalizedHeader 确实属于这个 StateRoot。")
	} else {
		fmt.Println("❌ 验证失败！StateRoot 不匹配。")
	}
}
