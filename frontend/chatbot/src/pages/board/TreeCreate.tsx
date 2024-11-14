import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";
import { addNode, saveNodeEdit, startEditNode, TreeNode, updateEditNodeData } from "@app/slices/treeSlice";
import { useAppDispatch, useAppSelector } from "@app/hooks";

const TreeCreate = () => {
	const {treeData, editNodeId, editNodeData} = useAppSelector(state=>state.tree);
	const dispatch = useAppDispatch();

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
		dispatch(startEditNode({
			id:node.id, 
			title: node.title || "", 
			content: node.content || ""
		}))
	};

	const handleSaveEdit = () => {
		dispatch(saveNodeEdit());
	};

	const handleAddNode = (parentId: string | null, depth: number) => {
		dispatch(addNode({parentId, parentDepth:depth}));
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
								value={editNodeData?.title}
								onChange={(e) => {
									dispatch(updateEditNodeData({title: e.target.value}))
								}}
								placeholder="제목을 입력하세요"
								className="px-2 py-1 border rounded"
								onClick={(e) => e.stopPropagation()}
							/>
							<input
								type="text"
								value={editNodeData?.content}
								onChange={(e) =>
									dispatch(updateEditNodeData({content: e.target.value}))
								}
								placeholder="내용을 입력하세요"
								className="px-2 py-1 border rounded"
								onClick={(e) => e.stopPropagation()}
							/>
							<button
								onClick={(e) => {
									e.stopPropagation();
									handleSaveEdit();
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
			</div>
		</div>
	);
};

export default TreeCreate;
