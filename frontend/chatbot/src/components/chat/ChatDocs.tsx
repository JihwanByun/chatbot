import React from 'react';
import CardWithModal from '@components/atoms/card/CardWithModal';
import { Doc } from './ChatView';

interface ChatDocsProps {
  docs: Doc[];
}

const EmptyDocsMessage = () => (
  <div className='bg-gray-50 rounded-lg p-8 text-center'>
    <div className='mb-6'>
      <svg 
        className="mx-auto h-12 w-12 text-gray-400"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    </div>
    <div className='text-gray-900 font-medium text-lg mb-2'>
      질문에 해당하는 자료가 없습니다.
    </div>
    <div className='text-gray-500'>
      sembot은 실수를 할 수 있습니다.
    </div>
  </div>
);

interface DocGridProps {
  docs: Doc[];
}

const DocGrid: React.FC<DocGridProps> = ({ docs }) => (
  <div className='grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-1 mx-auto w-full'>
    {docs.map((doc, index) => (
      <div className='w-full p-0.5'>
        <CardWithModal 
          key={index} 
          title={doc.metadata.filename} 
          content={doc.content}
        />
      </div>
    ))}
  </div>
);

const ChatDocs: React.FC<ChatDocsProps> = ({ docs }) => {
  return (
    <div className='container space-y-2'>
      <div className='text-lg font-bold mb-4'>
        출처
      </div>
      {!docs?.length ? <EmptyDocsMessage /> : <DocGrid docs={docs} />}
    </div>
  );
};

export default ChatDocs;