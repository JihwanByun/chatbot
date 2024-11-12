import React from 'react';
import CardWithModal from '@components/atoms/card/CardWithModal';
import { Doc } from './ChatView';

interface ChatDocsProps {
  docs: Doc[];
}

const ChatDocs: React.FC<ChatDocsProps> = ({ 
  docs,
}) => {
  return (
    <div className='container space-y-2'>
      <div className='text-lg font-bold'>
        출처
      </div>
      <div className='grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-2 place-items-center justify-center'>
        {docs.map((doc, index)=>(
          <CardWithModal key={index} title={doc.metadata.filename} content={doc.content}/>
        ))}
      </div>
    </div>
  );
};

export default ChatDocs;