import React, { useState } from "react";
import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";
import { useAppDispatch } from "@app/hooks";
import { createTree, updateTreeData } from "@app/slices/treeSlice";
import { TreeNode } from "@app/slices/treeSlice";

const TreeCreate = () => {
	const dispatch = useAppDispatch();
	const [treeData, setTreeData] = useState<TreeNode[]>([]);
	const [editNodeId, setEditNodeId] = useState<string | null>(null);
	const [editNodeData, setEditNodeData] = useState<{
		title: string;
		content: string;
	}>({
		title: "",
		content: "",
	});

	const getFormattedTitle = (
		depth: number,
		index: number,
		title?: string | null,
		content?: string | null
	) => {
		const depthLabel = ["장", "조", "항", "호"];
		const depthName = depthLabel[depth - 1] || "";
		return `제${index + 1}${depthName}(${title || ""}) ${content || ""}`;
	};

	const handleStartEdit = (node: TreeNode) => {
		setEditNodeId(node.id);
		setEditNodeData({
			title: node.title || "",
			content: node.content || "",
		});
	};

	const handleSaveEdit = (nodeId: string) => {
		setTreeData((prevData) =>
			updateNodeInTree(prevData, nodeId, {
				title: editNodeData.title,
				content: editNodeData.content,
			})
		);
		setEditNodeId(null);
		setEditNodeData({ title: "", content: "" });
	};

	const updateNodeInTree = (
		nodes: TreeNode[],
		nodeId: string,
		newData: { title: string; content: string }
	): TreeNode[] => {
		return nodes.map((node) => {
			if (node.id === nodeId) {
				return { ...node, ...newData };
			}
			if (node.children) {
				return {
					...node,
					children: updateNodeInTree(node.children, nodeId, newData),
				};
			}
			return node;
		});
	};

	const handleAddNode = (parentId: string | null, depth: number) => {
		const newNode: TreeNode = {
			id: `node-${Date.now()}`,
			title: "",
			content: "",
			children: [],
			depth: depth + 1,
		};

		if (parentId === null) {
			setTreeData([...treeData, newNode]);
		} else {
			setTreeData((prevData) => addChildNode(prevData, parentId, newNode));
		}
		handleStartEdit(newNode);
	};

	const addChildNode = (
		nodes: TreeNode[],
		parentId: string,
		newNode: TreeNode
	): TreeNode[] => {
		return nodes.map((node) => {
			if (node.id === parentId) {
				return {
					...node,
					children: [...(node.children || []), newNode],
				};
			}
			if (node.children) {
				return {
					...node,
					children: addChildNode(node.children, parentId, newNode),
				};
			}
			return node;
		});
	};

	// handleSaveTree 함수 수정
	const handleSaveTree = async () => {
		try {
			// 먼저 로컬 트리 데이터를 리덕스 스토어에 저장
			dispatch(updateTreeData(treeData));

			// createTree 액션 디스패치
			const result = await dispatch(createTree()).unwrap();

			// 성공 처리
			console.log("Tree saved successfully:", result);
			// 여기에 성공 메시지나 리다이렉션 로직 추가
		} catch (error) {
			console.error("Failed to save tree:", error);
			// 에러 메시지 표시 로직 추가
		}
	};

	const renderTree = (node: TreeNode, index: number) => (
		<TreeItem
			key={node.id}
			itemId={node.id}
			label={
				<div className="flex items-center w-full py-2 px-4 hover:bg-gray-50 justify-between">
					{node.id === editNodeId ? (
						<div className="flex gap-2 items-center">
							<input
								type="text"
								value={editNodeData.title}
								onChange={(e) =>
									setEditNodeData((prev) => ({
										...prev,
										title: e.target.value,
									}))
								}
								placeholder="제목을 입력하세요"
								className="px-2 py-1 border rounded"
								onClick={(e) => e.stopPropagation()}
							/>
							<input
								type="text"
								value={editNodeData.content}
								onChange={(e) =>
									setEditNodeData((prev) => ({
										...prev,
										content: e.target.value,
									}))
								}
								placeholder="내용을 입력하세요"
								className="px-2 py-1 border rounded"
								onClick={(e) => e.stopPropagation()}
							/>
							<button
								onClick={(e) => {
									e.stopPropagation();
									handleSaveEdit(node.id);
								}}
							>
								<CheckIcon fontSize="small" />
							</button>
						</div>
					) : (
						<div className="flex items-center justify-between w-full">
							<span>
								{getFormattedTitle(node.depth, index, node.title, node.content)}
							</span>
							<div className="flex gap-2">
								<button
									onClick={(e) => {
										e.stopPropagation();
										handleStartEdit(node);
									}}
								>
									<EditIcon fontSize="small" />
								</button>
								{node.depth <= 3 && (
									<button
										onClick={(e) => {
											e.stopPropagation();
											handleAddNode(node.id, node.depth);
										}}
									>
										<AddIcon fontSize="small" />
									</button>
								)}
							</div>
						</div>
					)}
				</div>
			}
		>
			{node.children?.map((child, i) => renderTree(child, i))}
		</TreeItem>
	);

	return (
		<div className="p-4">
			<SimpleTreeView>
				{treeData.map((node, index) => renderTree(node, index))}
			</SimpleTreeView>

			<div className="flex justify-between mt-4">
				<button
					className="bg-blue-500 text-white px-4 py-2 rounded flex items-center"
					onClick={() => handleAddNode(null, 0)}
				>
					<AddIcon fontSize="small" className="mr-2" />장 추가
				</button>

				<button
					className="bg-green-500 text-white px-4 py-2 rounded"
					onClick={handleSaveTree}
				>
					저장하기
				</button>
			</div>
		</div>
	);
};

export default TreeCreate;
