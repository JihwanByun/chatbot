import defaultAxios from "@apis/common";
import { TableRowData } from "@components/atoms/table/TableWithIcon";
import { favoritePath, favoritedPath } from "@pages/board";

//===== Backend API =====//

/**
 * 게시판 목록 조회 API 응답 타입
 * - contents: 게시글 목록 데이터
 * - page: 현재 페이지 번호 (0부터 시작)
 * - size: 페이지당 게시글 수
 * - totalPages: 전체 페이지 수
 * - totalElements: 전체 게시글 수
 */
export type BoardListResponse = {
	contents: BoardResponse[];
	page: number;
	size: number;
	totalPages: number;
	totalElements: number;
};

/**
 * 개별 게시글 데이터 타입
 * - boardId: 게시글 고유 식별자
 * - title: 게시글 제목
 * - contents: 게시글 내용
 * - createdAt: 작성일시
 * - name: 작성자 이름
 * - isFavorite: 즐겨찾기 여부
 */
export type BoardResponse = {
	boardId: number;
	title: string;
	contents: string;
	createdAt: string;
	name: string;
	isFavorite: boolean;
};

/**
 * 게시글 검색 조건 타입
 * - level?: 게시글 레벨 (선택)
 * - name?: 작성자 이름으로 검색 (선택)
 * - title?: 제목으로 검색 (선택)
 */
export type BoardSearchCondition = {
	level?: number;
	name?: string;
	title?: string;
};

// interface TreeState {
// 	treeData: TreeNode[];
// 	isRevisionMode: boolean;
// 	editNodeId: string | null;
// 	editNodeData: {
// 		title: string;
// 		content: string;
// 	} | null;
// }

// const initialState: TreeState = {
// 	isRevisionMode: false,
// 	editNodeId: null,
// 	editNodeData: null,
// 	treeData: [],
// };

export interface TreeNode {
	id: string;
	title?: string | null;
	content?: string | null;
	children?: TreeNode[];
	depth: number;
}

/**
 * 페이지네이션 정보 타입
 * - page: 요청할 페이지 번호 (0부터 시작)
 * - size: 한 페이지당 표시할 항목 수
 * - sort: 정렬 기준 배열 (예: ["createdAt,desc"])
 */
export type Pageable = {
	page: number;
	size: number;
	sort: string[];
};

/**
 * 테이블 컴포넌트에 표시할 데이터 타입
 * BoardListResponse를 테이블 형식으로 변환한 타입
 * - contents: 테이블에 표시될 행 데이터
 * - iconPaths: 각 행의 즐겨찾기 아이콘 경로
 * - page, size, totalPages, totalElements: 페이지네이션 정보
 */
export type TableResponse = {
	contents: TableRowData[];
	iconPaths: { [key: number]: string };
	page: number;
	size: number;
	totalPages: number;
	totalElements: number;
};

/**
 * API 응답을 테이블 형식으로 변환하는 함수
 * @param res - BoardListResponse 타입의 API 응답 데이터
 * @returns TableResponse 타입으로 변환된 데이터
 */
const convertToTable = (res: BoardListResponse): TableResponse => {
	return {
		...res,
		// 게시글 목록을 테이블 행 데이터로 변환
		contents: convertToTableRowList(res.contents),
		// 즐겨찾기 상태에 따른 아이콘 경로 매핑
		iconPaths: res.contents.reduce(
			(acc, content) => ({
				...acc,
				[content.boardId]: content.isFavorite ? favoritedPath : favoritePath,
			}),
			{}
		),
	};
};

/**
 * 단일 게시글 데이터를 테이블 행 데이터로 변환하는 함수
 * @param res - BoardResponse 타입의 단일 게시글 데이터
 * @returns TableRowData 타입의 테이블 행 데이터
 */
const convertToTableRow = (res: BoardResponse): TableRowData => {
	return {
		id: res.boardId,
		columns: [res.name, res.title, res.createdAt], // 작성자, 제목, 작성일 순서로 표시
	};
};

/**
 * 게시글 목록을 테이블 행 데이터 배열로 변환하는 함수
 * @param response - BoardResponse[] 타입의 게시글 목록
 * @returns TableRowData[] 타입의 테이블 행 데이터 배열
 */
const convertToTableRowList = (response: BoardResponse[]): TableRowData[] => {
	return response.map(convertToTableRow);
};

/**
 * 게시글 목록을 조회하는 API 함수
 * @param condition - 검색 조건 (선택)
 * @param pageInfo - 페이지네이션 정보 (선택)
 * @returns Promise<TableResponse | null> - 변환된 테이블 데이터 또는 에러 시 null
 */
export const getBoardListAPI = async (
	condition?: BoardSearchCondition,
	pageInfo?: Pageable
): Promise<TableResponse | null> => {
	return defaultAxios
		.get<BoardListResponse>("/api/boards", {
			params: {
				...condition,
				...pageInfo,
				sort: pageInfo?.sort.join(","), // 정렬 조건을 쉼표로 구분된 문자열로 변환
			},
		})
		.then((res) => {
			return convertToTable(res.data);
		})
		.catch((error) => {
			console.error("Error in boardListAPI:", error);
			return null;
		});
};

export type BoardUpdateRequestDto = {
	title?: string;
	contents?: string;
	category: string;
	level: number;
	regulationRequest?: RegulationRequestDto;
	hasFile?: boolean;
};

export type BoardCreateRequestDto = {
	title?: string;
	contents?: string;
	category: string;
	level: number;
	regulationRequest?: RegulationRequestDto;
	hasFile?: boolean;
};

export type RegulationRequestDto = {
	itemList: RegulationItemRequestDto[];
};

export type RegulationItemRequestDto = {
	title: string | null;
	content: string | null;
	itemList?: RegulationItemRequestDto[];
};

export interface TreeNode {
	id: string;
	title?: string | null;
	content?: string | null;
	children?: TreeNode[];
	depth: number;
}

function convertTreeNodeToRegulationItem(
	treeNode: TreeNode
): RegulationItemRequestDto {
	return {
		title: treeNode.title || null,
		content: treeNode.content || null,
		itemList: treeNode.children
			? treeNode.children.map(convertTreeNodeToRegulationItem)
			: undefined,
	};
}

function convertTreeDataToRegulationRequest(
	treeData: TreeNode[]
): RegulationRequestDto {
	return {
		itemList: treeData.map(convertTreeNodeToRegulationItem),
	};
}

export const updateBoard = async (
	boardId: string | undefined,
	treeData: TreeNode[],
	file?: File | null
) => {
	// FormData 객체 생성
	const formData = new FormData();

	// treeData를 RegulationRequestDto 구조로 변환
	const regulationRequestDto = convertTreeDataToRegulationRequest(treeData);

	// BoardUpdateRequestDto 구조 생성
	const boardUpdateRequestDto: BoardUpdateRequestDto = {
		title: "제목제목제목",
		contents: "내용내용내용",
		category: "휴가 지침",
		level: 3,
		regulationRequest: regulationRequestDto,
		hasFile: !!file,
	};

	// request 부분을 JSON 문자열로 변환하여 FormData에 추가
	formData.append(
		"request",
		new Blob([JSON.stringify(boardUpdateRequestDto)], {
			type: "application/json",
		})
	);

	// 파일이 있는 경우 추가
	if (file) {
		formData.append("file", file);
	}

	// API 요청 전송
	return defaultAxios.put(`/api/boards/${boardId}`, formData, {
		headers: {
			"Content-Type": "multipart/form-data",
		},
	});
};

export type BoardRequest = {
	title: string;
	category: string;
	level: number;
	file?: File;
}

// boardApi.ts
export const createBoard = async (treeData: TreeNode[], request : BoardRequest) => {
	// FormData 객체 생성
	console.log("Creating board with treeData:", treeData);

	const formData = new FormData();

	// treeData를 RegulationRequestDto 구조로 변환
	const regulationRequestDto = convertTreeDataToRegulationRequest(treeData);

	const file = request.file;

	// BoardUpdateRequestDto 구조 생성
	const boardCreateRequestDto: BoardCreateRequestDto = {
		title : request.title,
		category : request.category,
		level : request.level,
		regulationRequest: regulationRequestDto,
		hasFile: file ? true : false,
	};

	console.log("Request DTO:", boardCreateRequestDto);

	// request 부분을 JSON 문자열로 변환하여 FormData에 추가
	formData.append(
		"request",
		new Blob([JSON.stringify(boardCreateRequestDto)], {
			type: "application/json",
		})
	);

	// 파일이 있는 경우 추가
	if (file) {
		formData.append("file", file);
	}

	// FormData 내용 로깅
	for (const pair of formData.entries()) {
		console.log(pair[0], pair[1]);
	}

	// API 요청 전송 - 경로 수정
	const response = await defaultAxios.post("/api/boards", formData, {
		headers: {
			"Content-Type": "multipart/form-data",
		},
	});

	console.log("API Response:", response);
	return response;
};
