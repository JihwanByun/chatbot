import { createAsyncThunk, createSlice, PayloadAction } from "@reduxjs/toolkit";
import { RegulationResponse, RegulationItem } from "@apis/board/boardDetailApi";
import { createBoard, updateBoard } from "@apis/board/boardApi";

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
	isRevisionMode: false,
	editNodeId: null,
	editNodeData: null,
	treeData: [],
};

// RegulationItem을 TreeNode로 변환하는 헬퍼 함수
const convertToTreeNode = (
	item: RegulationItem,
	depth: number = 1
): TreeNode => ({
	...item,
	id: Math.random().toString(36).substr(2, 9), // 임시 ID 생성
	depth,
	children:
		item.itemList?.map((child) => convertToTreeNode(child, depth + 1)) || [],
});

export const saveTreeChange = createAsyncThunk(
	"board/update",
	async ({ boardId }: { boardId: string | undefined }, { getState }) => {
		// Access the treeData state
		const state = getState() as { tree: TreeState };
		const { treeData } = state.tree;

		// Call the API with treeData and boardId
		const response = await updateBoard(boardId, treeData);
		console.log("API Response:", response.data);

		return response.data;
	}
);

export const createTree = createAsyncThunk(
	"board/create",
	async (_, { getState }) => {
		// Access the treeData state
		const state = getState() as { tree: TreeState };
		const { treeData } = state.tree;

		// Call the API with treeData and boardId
		const response = await createBoard(treeData);
		console.log("API Response:", response.data);

		return response.data;
	}
);

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
			console.log(state.treeData);
		},
		cancelEdit: (state) => {
			state.editNodeId = null;
			state.editNodeData = null;
			state.isRevisionMode = false;
		},
		setTreeData: (state, action: PayloadAction<RegulationResponse | null>) => {
			if (!action.payload) {
				state.treeData = [];
				return;
			}
			state.treeData = action.payload.itemList.map((item) =>
				convertToTreeNode(item)
			);
		},
		updateTreeData: (state, action: PayloadAction<TreeNode[]>) => {
			state.treeData = action.payload;
			console.log(state.treeData);
			createTree();
		},
	},
	extraReducers: (builder) => {
		builder
			.addCase(createTree.pending, (state) => {
				// 필요한 경우 로딩 상태 처리
			})
			.addCase(createTree.fulfilled, (state, action) => {
				// API 응답 처리
				console.log("Tree created successfully:", action.payload);
			})
			.addCase(createTree.rejected, (state, action) => {
				// 에러 처리
				console.error("Failed to create tree:", action.error);
			});
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
	setTreeData,
	updateTreeData, // 새로운 액션 export
} = treeSlice.actions;

export default treeSlice.reducer;
