import React, { useCallback, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import TreeView from "@pages/board/TreeView";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import {
  setRevisionMode,
  startEditNode,
  saveNodeEdit,
  cancelEdit,
} from "@app/slices/treeSlice";
import {
  getBoardDetailAPI,
  downloadFileAPI,
  BoardDetailResponse,
} from "@apis/board/boardDetailApi";
import {
  createFavoriteAPI,
  deleteFavoriteAPI,
} from "@apis/board/boardFavoriteApi";
import ReactMarkdown from "react-markdown";
import { setTreeData } from "@app/slices/treeSlice";

interface BoardParams {
  id: string;
  [key: string]: string | undefined;
}

const BoardDetailPage: React.FC = () => {
  const { id } = useParams<BoardParams>();
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const [boardDetail, setBoardDetail] = useState<BoardDetailResponse | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [role, setRole] = useState<string | null>("");

  // Redux state
  const isRevisionMode = useAppSelector((state) => state.tree.isRevisionMode);
  const editNodeData = useAppSelector((state) => state.tree.editNodeData);

  // Role 확인
  useEffect(() => {
    const storedRole = localStorage.getItem("Role");
    if (storedRole) setRole(storedRole);
  }, []);

  // 게시글 상세 정보 조회
  const fetchBoardDetail = useCallback(async () => {
    if (!id) {
      setError("Invalid board ID");
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      const response = await getBoardDetailAPI(Number(id));
      setBoardDetail(response);

      // 규정 데이터를 Redux store에 설정
      dispatch(setTreeData(response.regulationResponseDto));
    } catch (error) {
      console.error("Failed to fetch board detail:", error);
      setError("게시글을 불러오는데 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  }, [id, dispatch]);

  // 파일 다운로드 처리
  const handleDownloadFile = async () => {
    if (!boardDetail?.fileUrl) return;

    try {
      const blob = await downloadFileAPI(boardDetail.fileUrl);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const fileName = boardDetail.fileUrl.split("/").pop() || "download";
      link.download = fileName;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download file:", error);
    }
  };

  // 즐겨찾기 토글
  const handleFavoriteToggle = async () => {
    if (!boardDetail) return;

    try {
      const success = boardDetail.isFavorite
        ? await deleteFavoriteAPI(boardDetail.boardId)
        : await createFavoriteAPI(boardDetail.boardId);

      if (success) {
        await fetchBoardDetail();
      }
    } catch (error) {
      console.error("Failed to toggle favorite:", error);
    }
  };

  // TreeView 관련 함수들
  const toggleRevisionMode = () => dispatch(setRevisionMode(!isRevisionMode));
  const discardEdit = () => dispatch(cancelEdit());

  const handleSaveEdit = (nodeId: string) => {
    if (editNodeData) {
      dispatch(
        saveNodeEdit({
          id: nodeId,
          title: editNodeData.title || "",
          content: editNodeData.content || "",
        })
      );
    }
  };

  useEffect(() => {
    fetchBoardDetail();
  }, [fetchBoardDetail]);

  const getChildren = () => (
    <div className="bg-white rounded-lg px-6 space-y-6 text-left">
      {/* 상단 버튼 영역 */}
      <div className="flex items-center space-x-4">
        <ButtonOnlyIcon
          key="move-prev-board"
          icon="/src/assets/icons/go-to-prev.svg"
          styleName="p-2 hover:bg-gray-100 rounded"
          onClick={() => navigate(-1)}
        />
        <ButtonOnlyIcon
          key="favorite"
          icon={
            boardDetail?.isFavorite
              ? "/src/assets/icons/favorited.svg"
              : "/src/assets/icons/favorite.svg"
          }
          width="20rem"
          styleName="p-2 hover:bg-gray-100 rounded"
          onClick={handleFavoriteToggle}
        />
        {role === "일반 사용자 작성자" ? (
          !isRevisionMode ? (
            <ButtonOnlyIcon
              key="edit"
              icon="../src/assets/icons/pen.svg"
              styleName="p-2 hover:bg-gray-100 rounded ml-4"
              width="20rem"
              onClick={toggleRevisionMode}
            />
          ) : (
            <div className="flex">
              <ButtonOnlyIcon
                key="save"
                icon="../src/assets/icons/save.svg"
                styleName="p-2 hover:bg-gray-100 rounded"
                width="20rem"
                onClick={toggleRevisionMode}
              />
              <ButtonOnlyIcon
                key="discard"
                icon="../src/assets/icons/x-circle.svg"
                styleName="p-2 hover:bg-gray-100 rounded ml-2"
                width="20rem"
                onClick={discardEdit}
              />
            </div>
          )
        ) : null}
      </div>

      {/* 게시글 헤더 */}
      <div className="space-y-4">
        <div className="flex items-start gap-4">
          <h1 className="text-xl font-semibold">{boardDetail?.title}</h1>
          <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 whitespace-nowrap">
            답변 레벨 : {boardDetail?.level}
          </span>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div className="flex items-center space-x-2">
            <img
              className="h-8 w-8 rounded-full"
              src={boardDetail?.writer.profileUrl}
              alt={boardDetail?.writer.name}
            />
            <span className="font-medium">{boardDetail?.writer.name}</span>
            <span className="text-gray-500 text-xs">
              &lt;{boardDetail?.writer.email}&gt;
            </span>
          </div>
          <div className="flex flex-col items-start sm:items-end">
            <span className="text-xs text-gray-500">
              {boardDetail && new Date(boardDetail.createdAt).toLocaleString()}
            </span>
            {boardDetail?.hasFile && (
              <ButtonWithIcon
                btnName="파일 다운로드"
                styleName="mt-2 text-sm text-blue-600 hover:text-blue-800"
                icon="/src/assets/icons/document-download.svg"
                handleClick={handleDownloadFile}
              />
            )}
          </div>
        </div>
      </div>

      <hr className="border-gray-200" />

      {/* 게시글 내용 - Markdown 렌더링 */}
      <div
        className="prose max-w-none text-left
                [&>*]:text-left 
                prose-headings:text-left
                prose-p:text-left
                prose-ul:text-left
                prose-ol:text-left"
      >
        {/* 게시글 내용 - Markdown 렌더링 */}
        {boardDetail?.contents?.replace(/\\n/g, "\n") && (
          <ReactMarkdown>
            {boardDetail.contents.replace(/\\n/g, "\n")}
          </ReactMarkdown>
        )}
      </div>

      {/* 규정 트리뷰 */}
      {boardDetail?.regulationResponseDto && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-4">규정 정보</h3>
          <TreeView
            isRevisionMode={isRevisionMode}
            handleStartEdit={(node) =>
              dispatch(
                startEditNode({
                  id: node.id,
                  title: node.title || "",
                  content: node.content || "",
                })
              )
            }
            handleSaveEdit={handleSaveEdit}
          />
        </div>
      )}
    </div>
  );

  const sidebarComponents = [
    {
      btnName: "규정목록",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/book-open-text.svg",
    },
    {
      btnName: boardDetail?.title || "규정",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/book-open-text.svg",
    },
  ];

  const footerComponents = [
    {
      btnName: "채팅",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/chatting-icon.svg",
    },
    {
      btnName: "규정 확인하기",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/book-open-text.svg",
    },
    {
      btnName: "로그아웃",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/logout.svg",
    },
  ];

  if (error) {
    return (
      <SembotLayout
        title="오류"
        sidebarComponents={sidebarComponents}
        footerComponents={footerComponents}
      >
        <div className="flex justify-center items-center h-full">
          <div className="text-red-500">{error}</div>
        </div>
      </SembotLayout>
    );
  }

  if (isLoading) {
    return (
      <SembotLayout
        title="로딩 중..."
        sidebarComponents={sidebarComponents}
        footerComponents={footerComponents}
      >
        <div className="flex justify-center items-center h-full">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      </SembotLayout>
    );
  }

  return (
    <SembotLayout
      title={boardDetail?.title || "규정 상세"}
      sidebarComponents={sidebarComponents}
      footerComponents={footerComponents}
      children={getChildren()}
    />
  );
};

export default BoardDetailPage;
