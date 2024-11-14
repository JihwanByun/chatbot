import Swal from "sweetalert2";

export const deleteAlert = (handleClick : () => void) => {
  Swal.fire({
    title: "삭제하시겠습니까?",
    text: "삭제 후 복구는 불가능합니다.",
    icon: "warning",
    showCancelButton: true,
    // confirmButtonColor: '#d33',
    // cancelButtonColor: '#3085d6',
    confirmButtonText: "삭제",
    cancelButtonText: "취소",
    customClass: {
      confirmButton:
        "bg-red-500 text-white px-4 py-2 rounded-lg focus:ring-0 focus:outline-none active:bg-red-500 hover:bg-red-600",
      cancelButton:
        "bg-white border-2 border-gray-300 text-gray-700 px-4 py-2 rounded-lg ml-2 hover:bg-gray-200",
    },
    buttonsStyling: false, // 기본 스타일링 비활성화
  }).then((result) => {
    if (result.isConfirmed) {
      handleClick();
      Swal.fire("삭제되었습니다.");
    }
  });
}

export const errorAlert = (error : Error) => {
  Swal.fire({
    text: error.message || '오류가 발생했습니다.',
    icon: 'error',
    confirmButtonText: '확인',
    customClass: {
      confirmButton:
        'bg-blue-500 text-white px-4 py-2 rounded-lg focus:ring-0 focus:outline-none active:bg-blue-500 hover:bg-blue-600',
    },
    buttonsStyling: false,
  });
}

export const successAlert = (message? : string) => {
  Swal.fire({
    text: message || '완료되었습니다.',
    icon: 'success',
    confirmButtonText: '확인',
    customClass: {
      confirmButton:
      'bg-blue-500 text-white px-4 py-2 rounded-lg focus:ring-0 focus:outline-none active:bg-blue-500 hover:bg-blue-600',
    },
    buttonsStyling: false,
  });
}