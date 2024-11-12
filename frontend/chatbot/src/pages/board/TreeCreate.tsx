import React, { useState } from "react";
import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import AddIcon from "@mui/icons-material/Add";
import CheckIcon from "@mui/icons-material/Check";
import { useAppDispatch } from "@app/hooks";
import { addNode } from "@app/slices/treeSlice";
import { TreeNode } from "@app/slices/treeSlice";

const TreeCreate = () => {
	const dispatch = useAppDispatch();
	const [treeData, setTreeData] = useState<TreeNode[]>([]);
	const [newNodeData, setNewNodeData] = useState<{
		title: string;
		content: string;
	}>({
		title: "",
		content: "",
	});

	const handleChange = (field: "title" | "content", value: string) => {
		setNewNodeData((prevData) => ({ ...prevData, [field]: value }));
	};

	const handleAddNode = () => {
		if (!newNodeData.title || !newNodeData.content) return;

		const newNode: TreeNode = {
			id: `node-${Date.now()}`,
			title: newNodeData.title,
			content: newNodeData.content,
			children: [],
			depth: 1,
		};

		setTreeData([...treeData, newNode]);
		setNewNodeData({ title: "", content: "" });

		// 필요시 store에도 추가
		dispatch(addNode({ parentId: null, parentDepth: 0 }));
	};

	const renderTree = (node: TreeNode) => (
		<TreeItem
			key={node.id}
			itemId={node.id}
			label={
				<div className="flex items-center w-full py-2 px-4 hover:bg-gray-50 justify-between">
					<span>{node.title}</span>
				</div>
			}
		>
			{node.children?.map(renderTree)}
		</TreeItem>
	);

	return (
		<div>
			<div className="flex items-center w-full py-2 px-4">
				<input
					type="text"
					value={newNodeData.title}
					onChange={(e) => handleChange("title", e.target.value)}
					placeholder="제목을 입력하세요."
					className="px-2 py-1 border rounded flex-grow"
				/>
				<input
					type="text"
					value={newNodeData.content}
					onChange={(e) => handleChange("content", e.target.value)}
					placeholder="내용을 입력하세요."
					className="px-2 py-1 border rounded flex-grow ml-2"
				/>
				<button onClick={handleAddNode} className="ml-2">
					<CheckIcon fontSize="small" />
				</button>
			</div>

			<SimpleTreeView>
				{treeData.map((node) => renderTree(node))}
			</SimpleTreeView>

			<div className="flex justify-center mt-4">
				<button
					className="bg-blue-500 text-white px-4 py-2 rounded flex items-center "
					onClick={() =>
						dispatch(
							addNode({
								parentId: null,
								parentDepth: 0, // 최상위 장 추가
							})
						)
					}
				>
					<AddIcon fontSize="small" className="mr-2 " />장 추가
				</button>
			</div>
		</div>
	);
};

export default TreeCreate;
