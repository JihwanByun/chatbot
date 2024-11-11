import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface TreeNode {
	id: string;
	title?: string | null;
	content?: string | null;
	children?: TreeNode[];
	depth: number;
}

interface TreeState {
	treeData: TreeNode[];
	isRevisionMode: boolean;
	editNodeId: string | null;
	editNodeData: {
		title: string;
		content: string;
	} | null;
}

const initialState: TreeState = {
	treeData: [
		{
			id: "1",
			content: `휴가 지침`,
			title: "휴가",
			children: [
				{
					id: "2",
					title: "휴가의 범위",
					content: "휴가는 열심히 일한자만 갈 수 있습니다.",
					depth: 2,
				},
				{
					id: "3",
					title: "휴가의 종류",
					content: "휴가 따위 없다 일이나 해라",
					depth: 2,
				},
			],
			depth: 1,
		},
	],
	isRevisionMode: false,
	editNodeId: null,
	editNodeData: null,
};

const treeSlice = createSlice({
	name: "tree",
	initialState,
	reducers: {
		setRevisionMode: (state, action: PayloadAction<boolean>) => {
			console.log("편집 모드 on");
			state.isRevisionMode = action.payload;
			// 수정 모드를 끄면 편집 상태도 초기화
			if (!action.payload) {
				state.editNodeId = null;
				state.editNodeData = null;
			}
		},
		addNode: (
			state,
			action: PayloadAction<{ parentId: string | null; parentDepth: number }>
		) => {
			const { parentId, parentDepth } = action.payload;
			const newNode: TreeNode = {
				id: `node-${Date.now()}`,
				content: "",
				depth: parentDepth + 1,
				title: "",
			};

			if (!parentId) {
				state.treeData.push(newNode);
				return;
			}

			const updateTree = (nodes: TreeNode[]): TreeNode[] => {
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
							children: updateTree(node.children),
						};
					}
					return node;
				});
			};

			state.treeData = updateTree(state.treeData);
		},
		deleteNode: (state, action: PayloadAction<string>) => {
			const nodeId = action.payload;
			const deleteFromTree = (nodes: TreeNode[]): TreeNode[] => {
				return nodes.filter((node) => {
					if (node.id === nodeId) {
						return false;
					}
					if (node.children) {
						node.children = deleteFromTree(node.children);
					}
					return true;
				});
			};

			state.treeData = deleteFromTree(state.treeData);
			if (state.editNodeId === nodeId) {
				state.editNodeId = null;
				state.editNodeData = null;
			}
		},
		startEditNode: (
			state,
			action: PayloadAction<{ id: string; title: string; content: string }>
		) => {
			state.editNodeId = action.payload.id;
			state.editNodeData = {
				title: action.payload.title,
				content: action.payload.content,
			};
		},
		updateEditNodeData: (
			state,
			action: PayloadAction<{ title?: string; content?: string }>
		) => {
			if (state.editNodeData) {
				state.editNodeData = {
					...state.editNodeData,
					...action.payload,
				};
			}
		},
		saveNodeEdit: (
			state,
			action: PayloadAction<{ id: string; title: string; content: string }>
		) => {
			const { id, title, content } = action.payload;
			const updateTree = (nodes: TreeNode[]): TreeNode[] => {
				return nodes.map((node) => {
					if (node.id === id) {
						return {
							...node,
							title,
							content,
						};
					}
					if (node.children) {
						return {
							...node,
							children: updateTree(node.children),
						};
					}
					return node;
				});
			};

			state.treeData = updateTree(state.treeData);
			state.editNodeId = null;
			state.editNodeData = null;
		},
		cancelEdit: (state) => {
			state.editNodeId = null;
			state.editNodeData = null;
		},
	},
});

export const {
	setRevisionMode,
	addNode,
	deleteNode,
	startEditNode,
	updateEditNodeData,
	saveNodeEdit,
	cancelEdit,
} = treeSlice.actions;

export default treeSlice.reducer;
