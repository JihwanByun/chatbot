import React from "react";
import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";
import { useAppSelector, useAppDispatch } from "@app/hooks";
import { addNode, updateEditNodeData } from "@app/slices/treeSlice";
import { TreeNode } from "@app/slices/treeSlice";

interface TreeViewProps {
	isRevisionMode: boolean;
	handleStartEdit: (node: TreeNode) => void;
	handleSaveEdit: (nodeId: string) => void;
}

const TreeView: React.FC<TreeViewProps> = ({
	isRevisionMode,
	handleStartEdit,
	handleSaveEdit,
}) => {
	const dispatch = useAppDispatch();
	const { treeData, editNodeId, editNodeData } = useAppSelector(
		(state) => state.tree
	);

	const getFormattedTitle = (
		depth: number,
		index: number,
		title?: string | null,
		content?: string | null
	) => {
		const depthLabel = ["장", "조", "항", "호"];
		const depthName = depthLabel[depth - 1] || "";

		// title과 content가 모두 없는 경우 인덱스만 표시
		if (!title && !content) {
			return `제${index + 1}${depthName}`;
		}

		// title만 있는 경우
		if (title && !content) {
			return `제${index + 1}${depthName}(${title})`;
		}

		// content만 있는 경우
		if (!title && content) {
			return `제${index + 1}${depthName} ${content}`;
		}

		// 둘 다 있는 경우
		return `제${index + 1}${depthName}(${title}) ${content}`;
	};

	const renderTree = (node: TreeNode, index: number) => (
		<TreeItem
			key={node.id}
			itemId={node.id}
			label={
				<div className="flex items-center w-full py-2 px-4 hover:bg-gray-50 justify-between">
					{node.id === editNodeId && isRevisionMode ? (
						<div className="flex gap-2">
							<input
								type="text"
								value={editNodeData?.title || ""}
								onChange={(e) =>
									dispatch(updateEditNodeData({ title: e.target.value }))
								}
								placeholder="제목을 입력하세요."
								className="px-2 py-1 border rounded flex-grow"
								onClick={(e) => e.stopPropagation()}
							/>
							<input
								type="text"
								value={editNodeData?.content || ""}
								onChange={(e) =>
									dispatch(updateEditNodeData({ content: e.target.value }))
								}
								placeholder="내용을 입력하세요."
								className="px-2 py-1 border rounded flex-grow"
								onClick={(e) => e.stopPropagation()}
							/>
							<button
								onClick={(e) => {
									handleSaveEdit(node.id);
									e.stopPropagation();
								}}
							>
								<CheckIcon fontSize="small" />
							</button>
						</div>
					) : (
						<div className="flex justify-between w-full">
							<span className="flex-grow">
								{getFormattedTitle(node.depth, index, node.title, node.content)}
							</span>
							{isRevisionMode && (
								<div className="flex gap-2">
									<button
										onClick={(e) => {
											handleStartEdit(node);
											e.stopPropagation();
										}}
									>
										<EditIcon fontSize="small" />
									</button>
									{node.depth <= 3 && (
										<button
											onClick={(e) => {
												dispatch(
													addNode({
														parentId: node.id,
														parentDepth: node.depth,
													})
												);
												e.stopPropagation();
											}}
										>
											<AddIcon fontSize="small" />
										</button>
									)}
								</div>
							)}
						</div>
					)}
				</div>
			}
		>
			{node.children?.map((child, i) => renderTree(child, i))}
		</TreeItem>
	);

	return (
		<div>
			<SimpleTreeView>
				{treeData.map((node, index) => renderTree(node, index))}
			</SimpleTreeView>

			{isRevisionMode && (
				<div className="flex justify-center mt-4">
					<button
						className="bg-blue-500 text-white px-4 py-2 rounded flex items-center"
						onClick={() =>
							dispatch(
								addNode({
									parentId: null,
									parentDepth: 0,
								})
							)
						}
					>
						<AddIcon fontSize="small" className="mr-2" />장 추가
					</button>
				</div>
			)}
			<div className="flex h-[30px]"></div>
		</div>
	);
};

export default TreeView;